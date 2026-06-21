"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar Distance: hold a just-dropped, recently-sustained lead alive through radar flicker so the MPC does
not lose+regain it. Obstacle-monotone (held obstacle <= last real <= stock) -> braking is always >= stock,
never less. Wall-clock bounded, flicker-proof. Default off => stock passthrough.

Stop-neutral: at/below LOW_SPEED_PASSTHROUGH_V (the stop/creep regime) it returns the RAW radarstate
unchanged so the stop distance is byte-identical to stock (RadarDistance OFF). The flicker-hold only
governs above that speed, where lose+regain actually matters.
"""

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

HOLD_MAX_FRAMES = 10     # ~0.5s cap, measured since the last SUSTAINED lead (not reset by 1-frame flicker)
SUSTAIN_FRAMES = 2       # consecutive valid frames to (re)arm and reset the wall-clock
DROPOUT_DREL = 1.0
FCW_PROB_CAP = 0.9       # held lead can't reach the FCW gate (>0.9) -> no false FCW
MIN_HELD_DREL = 0.5

# Stop-neutrality gate. At/below this speed we are in the stop/creep regime, where the car's stop
# distance must match stock openpilot exactly (and radard's low_speed_override already supplies a robust
# closest-track lead). So below this speed smooth_radarstate returns the RAW radarstate unchanged --
# byte-identical to RadarDistance OFF -- so neither the flicker-hold's dead-reckoned dRel nor any other
# transform can move the lead the MPC sees near a stop. The hold is still stepped to keep its state warm
# for when speed rises back above the gate. Above this speed the highway flicker-hold runs (its purpose).
LOW_SPEED_PASSTHROUGH_V = 5.0  # m/s (~18 km/h): covers stop + creep, below highway following


class _HeldLead:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, dRel, vRel, vLead, aLeadK, aLeadTau, modelProb):
    self.status = True
    self.dRel = dRel
    self.vRel = vRel
    self.vLead = vLead
    self.vLeadK = vLead
    self.aLeadK = aLeadK
    self.aLeadTau = aLeadTau
    self.modelProb = modelProb
    self.yRel = 0.0


class _RadarStateProxy:
  __slots__ = ('leadOne', 'leadTwo')

  def __init__(self, lead_one, lead_two):
    self.leadOne = lead_one
    self.leadTwo = lead_two


class _LeadHold:
  def __init__(self):
    self._last = None
    self._sustained = 0
    self._since_real = 0
    self._armed = False
    self._held_dRel = 0.0

  def reset(self):
    self.__init__()

  def step(self, raw):
    # Validity mirrors the MPC, which keys off status alone (long_mpc process_lead). modelProb is NOT a
    # gate: radard's low_speed_override emits a real closest-track lead with modelProb=0.0, so gating on
    # prob wrongly rejected real close stop-and-go leads and substituted a stale farther held lead ->
    # under-brake -> stopping too close. FCW stays bounded by FCW_PROB_CAP on the held output below.
    if raw.status and raw.dRel > DROPOUT_DREL:
      self._last = (raw.dRel, raw.vRel, raw.vLead, raw.aLeadK, raw.aLeadTau, raw.modelProb)
      self._sustained += 1
      if self._sustained >= SUSTAIN_FRAMES:
        self._since_real = 0
        self._armed = True
      return raw

    self._sustained = 0
    self._since_real += 1
    if self._armed and self._last is not None and self._since_real <= HOLD_MAX_FRAMES:
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      if self._since_real == 1:
        self._held_dRel = dRel0
      self._held_dRel = max(MIN_HELD_DREL, self._held_dRel - max(-vRel0, 0.0) * DT_MDL)
      return _HeldLead(self._held_dRel, vRel0, vLead0, min(aLeadK0, 0.0), aLeadTau0, min(prob0, FCW_PROB_CAP))

    self._armed = False
    return raw


class RadarDistanceController:
  def __init__(self, CP: structs.CarParams, params=None):
    self._CP = CP
    self._params = params or Params()
    self._frame = 0
    self._v_ego = 0.0
    self._enabled = self._params.get_bool("RadarDistance")
    self._one = _LeadHold()
    self._two = _LeadHold()

  def _read_params(self) -> None:
    enabled = self._params.get_bool("RadarDistance")
    if enabled and not self._enabled:
      self._one.reset()
      self._two.reset()
    self._enabled = enabled

  def update(self, sm) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._read_params()
    self._v_ego = float(sm['carState'].vEgo)
    self._frame += 1

  def enabled(self) -> bool:
    return self._enabled

  def smooth_radarstate(self, radarstate):
    if not self._enabled:
      return radarstate
    # Step the holds every frame to keep their flicker state warm, but in the stop/creep regime return the
    # RAW radarstate unchanged so the lead the MPC sees -- and therefore the stop distance -- is byte-stock
    # (identical to RadarDistance OFF). Above the gate the highway flicker-hold governs (its real purpose).
    one = self._one.step(radarstate.leadOne)
    two = self._two.step(radarstate.leadTwo)
    if self._v_ego < LOW_SPEED_PASSTHROUGH_V:
      return radarstate
    return _RadarStateProxy(one, two)
