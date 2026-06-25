"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance: keep a just-dropped, recently-sustained lead alive through a brief radar dropout (flicker-hold),
so the MPC does not lose+regain a flickering lead. The held lead is obstacle-monotone (held obstacle <= last
real <= stock) -> braking is always >= stock, never weaker. Active only above LOW_SPEED_PASSTHROUGH_V; at/below
it (stop/creep) it returns the raw radarstate unchanged -> byte-stock stops. Default off => stock passthrough.

NOTE: an earlier vLead "rise smoothing" was removed -- it lagged the lead's speed-up by ~1 s, so when a lead
pulled away in stop-and-go it reported the lead as still near-stopped (measured up to 11 m/s slower than real).
That fed the MPC a phantom-slow/stopped lead -> phantom hard braking + a launch rubber-band. The lead's real
speed is passed through unchanged now.
"""

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

HOLD_MAX_FRAMES = 10        # ~0.5s flicker-hold cap, since the last sustained lead
SUSTAIN_FRAMES = 2          # consecutive valid frames to arm the hold
DROPOUT_DREL = 1.0
FCW_PROB_CAP = 0.9          # held lead can't reach the FCW gate (>0.9)
MIN_HELD_DREL = 0.5

# Stop/creep regime: return the raw radarstate so stop distance is byte-identical to stock (off==on).
LOW_SPEED_PASSTHROUGH_V = 5.0   # m/s


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
    # Validity mirrors the MPC (keys off status alone). modelProb is NOT a gate: radard's low_speed_override
    # emits a real close lead with modelProb=0.0, so gating on prob dropped real stop-and-go leads.
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
    one = self._one.step(radarstate.leadOne)
    two = self._two.step(radarstate.leadTwo)
    if self._v_ego < LOW_SPEED_PASSTHROUGH_V:               # stop/creep -> raw (byte-stock stops)
      return radarstate
    return _RadarStateProxy(one, two)                       # flicker-hold only; lead speed passed through as-is
