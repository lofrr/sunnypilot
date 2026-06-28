"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance smooths the lead the longitudinal MPC follows on a noisy radar, never reporting a
farther-or-faster lead than reality, so braking is always >= stock:
  - flicker-hold: keep a just-dropped, recently-sustained lead alive through a radar dropout.
  - speed damp: lag the lead speeding up (instant on slow-down) to damp the catch-up surge / rubber-band,
    reset on a track switch so it never carries a stale-slow value across a different track.
Active only above LOW_SPEED_PASSTHROUGH_V; at/below it returns the raw radarstate (byte-stock stops).
Default off => stock passthrough.
"""

from collections import deque

from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

HOLD_MAX_FRAMES = 20        # ~1.0s flicker-hold cap, since the last sustained lead
SUSTAIN_FRAMES = 2
DROPOUT_DREL = 1.0
FCW_PROB_CAP = 0.9
MIN_HELD_DREL = 0.5

LOW_SPEED_PASSTHROUGH_V = 5.0   # m/s

# Speed-damp (B) gated off (caused phantom braking + launch rubber-band before); flicker-hold (A) runs alone.
VLEAD_DAMP_ENABLED = False
VLEAD_TAU = 0.4                 # s, lag on a speeding-up lead
_VLEAD_ALPHA = DT_MDL / VLEAD_TAU
SWITCH_DREL = 8.0              # m, dRel jump that means the radar switched to a different track -> reset the filter

# Lead-instability detector (telemetry only): flags a bimodal/bouncing radar lead.
STABILITY_WINDOW = 5            # frames (~0.25s @ 20Hz)
VLEAD_SPREAD = 4.0             # m/s, vLead range over the window above which the lead is "unstable"
ID_CHURN_WINDOW = 10           # frames (~0.5s) for radarTrackId-churn detection (steady lead, flipping track ids)
ID_CHURN = 3                   # trackId switches in the window above which the lead is "unstable" (follow-hunting)

# Lead jitter smoother (B2): during trackId churn the per-track dRel/vRel jitter makes the MPC hunt the follow
# gap. A short SYMMETRIC EMA on the churning lead removes the jitter so the MPC sees a steady lead and stops
# hunting. Active ONLY during churn (NOT bimodal vLead -> never averages two real tracks). Bounded symmetric
# lag ~LEAD_SMOOTH_TAU. Gated OFF by default.
LEAD_SMOOTH_ENABLED = False
LEAD_SMOOTH_TAU = 0.5          # s, EMA time constant
LEAD_SMOOTH_HOLD = 20          # frames (~1s): keep smoothing through brief churn gaps (churn toggles on/off)

# Stop-gap bias: near a (near-)stopped lead at low speed, report dRel up to STOP_GAP_BIAS_M closer so the MPC
# runs its own smooth stop but terminates that much farther back (stock crawl-creeps to ~2m). Monotone (closer
# => brake >= stock). Ramps in over the regime edge and out as the lead moves (no step, releases on launch).
STOP_GAP_BIAS_ENABLED = False
STOP_GAP_BIAS_M = 2.0          # m: max dRel reduction = added standstill gap
STOP_BIAS_VEGO = 8.0           # m/s: only below this ego speed
STOP_BIAS_VLEAD = 1.5         # m/s: only behind a (near-)stopped lead; ramps out as vLead rises to this
STOP_BIAS_REGIME_DREL = 12.0   # m: bias ramps in below this dRel
STOP_BIAS_RAMP_BAND = 2.0     # m: ramp-in band (full offset below REGIME_DREL - RAMP_BAND)
STOP_BIAS_MIN_DREL = 2.0      # m: never report a lead closer than this


class _LeadView:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, src, vlead):
    self.status = src.status
    self.dRel = src.dRel
    self.yRel = src.yRel
    self.vRel = src.vRel
    self.vLead = vlead
    self.vLeadK = vlead
    self.aLeadK = src.aLeadK
    self.aLeadTau = src.aLeadTau
    self.modelProb = src.modelProb


class _BiasedLead:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, src, dRel):
    self.status = src.status
    self.dRel = dRel
    self.yRel = src.yRel
    self.vRel = src.vRel
    self.vLead = src.vLead
    self.vLeadK = src.vLeadK
    self.aLeadK = src.aLeadK
    self.aLeadTau = src.aLeadTau
    self.modelProb = src.modelProb


class _SmoothedLead:
  __slots__ = ('status', 'dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')

  def __init__(self, src, dRel, vLead, vRel):
    self.status = src.status
    self.dRel = dRel
    self.yRel = src.yRel
    self.vRel = vRel
    self.vLead = vLead
    self.vLeadK = vLead
    self.aLeadK = src.aLeadK
    self.aLeadTau = src.aLeadTau
    self.modelProb = src.modelProb


class _LeadSmoother:
  # Short symmetric EMA on a churning lead's dRel/vLead/vRel (jitter removal). A hold keeps it active through
  # brief churn gaps (churn toggles); passthrough + reset only after the hold lapses.
  def __init__(self):
    self._d = None
    self._vl = None
    self._vr = None
    self._hold = 0

  def reset(self):
    self._d = None
    self._vl = None
    self._vr = None
    self._hold = 0

  def update(self, lead, churn: bool):
    self._hold = LEAD_SMOOTH_HOLD if churn else self._hold - 1
    if self._hold <= 0 or not lead.status:
      self.reset()
      return lead
    if self._d is None:
      self._d, self._vl, self._vr = lead.dRel, lead.vLead, lead.vRel
      return lead
    a = DT_MDL / LEAD_SMOOTH_TAU
    self._d += (lead.dRel - self._d) * a
    self._vl += (lead.vLead - self._vl) * a
    self._vr += (lead.vRel - self._vr) * a
    return _SmoothedLead(lead, self._d, self._vl, self._vr)


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
    self._vlead_f = None
    self._last_dRel = None

  def reset(self):
    self.__init__()

  def step(self, raw):
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

  def smooth(self, lead):
    if not lead.status:
      self._vlead_f = None
      self._last_dRel = None
      return lead
    if self._last_dRel is None or abs(lead.dRel - self._last_dRel) > SWITCH_DREL:
      self._vlead_f = lead.vLead
    self._last_dRel = lead.dRel
    v = float(lead.vLead)
    if self._vlead_f is None or v <= self._vlead_f:
      self._vlead_f = v
      return lead
    self._vlead_f += (v - self._vlead_f) * _VLEAD_ALPHA
    return _LeadView(lead, self._vlead_f)


class _LeadStability:
  # Read-only monitor: flags an unstable leadOne -- bimodal/bouncing vLead, dRel track-switch jumps, or
  # radarTrackId churn (a steady lead flipping track ids -> vRel jitter -> follow-hunting). Telemetry only.
  def __init__(self):
    self._v = deque(maxlen=STABILITY_WINDOW)
    self._d = deque(maxlen=STABILITY_WINDOW)
    self._id = deque(maxlen=ID_CHURN_WINDOW)
    self.unstable = False
    self.churn = False

  def reset(self):
    self._v.clear()
    self._d.clear()
    self._id.clear()
    self.unstable = False
    self.churn = False

  def update(self, lead, v_ego: float) -> None:
    if not lead.status or v_ego < LOW_SPEED_PASSTHROUGH_V:
      self.reset()
      return
    self._v.append(float(lead.vLead))
    self._d.append(float(lead.dRel))
    self._id.append(int(getattr(lead, 'radarTrackId', -1)))
    if len(self._v) < STABILITY_WINDOW:
      self.unstable = False
      return
    v_spread = max(self._v) - min(self._v)
    d_jumps = sum(abs(b - a) > SWITCH_DREL for a, b in zip(self._d, list(self._d)[1:], strict=False))
    ids = list(self._id)
    id_churn = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a != b and a > 0 and b > 0)
    self.churn = id_churn >= ID_CHURN and v_spread <= VLEAD_SPREAD   # steady lead, flipping ids (not bimodal)
    self.unstable = v_spread > VLEAD_SPREAD or d_jumps >= 2 or self.churn


class RadarDistanceController:
  def __init__(self, CP: structs.CarParams, params=None):
    self._CP = CP
    self._params = params or Params()
    self._frame = 0
    self._v_ego = 0.0
    self._enabled = self._params.get_bool("RadarDistance")
    self._vlead_damp_enabled = VLEAD_DAMP_ENABLED
    self._stop_gap_bias_enabled = STOP_GAP_BIAS_ENABLED
    self._lead_smooth_enabled = LEAD_SMOOTH_ENABLED
    self._one = _LeadHold()
    self._two = _LeadHold()
    self._stability = _LeadStability()
    self._smoother = _LeadSmoother()

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

  def lead_unstable(self) -> bool:
    return self._stability.unstable

  def _stop_gap_bias(self, lead):
    # Report a (near-)stopped lead up to STOP_GAP_BIAS_M closer at low speed, so the MPC's own smooth stop ends
    # that much farther back. Monotone (only ever reports closer). No-op outside the regime / when disabled.
    if not self._stop_gap_bias_enabled or not lead.status:
      return lead
    if lead.vLead > STOP_BIAS_VLEAD or self._v_ego > STOP_BIAS_VEGO or lead.dRel <= STOP_BIAS_MIN_DREL:
      return lead
    d_ramp = min(max((STOP_BIAS_REGIME_DREL - lead.dRel) / STOP_BIAS_RAMP_BAND, 0.0), 1.0)
    v_ramp = min(max((STOP_BIAS_VLEAD - lead.vLead) / STOP_BIAS_VLEAD, 0.0), 1.0)
    offset = STOP_GAP_BIAS_M * d_ramp * v_ramp
    if offset < 0.05:
      return lead
    return _BiasedLead(lead, max(lead.dRel - offset, STOP_BIAS_MIN_DREL))

  def smooth_radarstate(self, radarstate):
    self._stability.update(radarstate.leadOne, self._v_ego)   # telemetry, runs every cycle
    if not self._enabled:
      return radarstate
    one = self._one.step(radarstate.leadOne)
    two = self._two.step(radarstate.leadTwo)
    if self._v_ego < LOW_SPEED_PASSTHROUGH_V:
      one_b = self._stop_gap_bias(radarstate.leadOne)         # low speed = stock lead, only the stop-gap bias
      return radarstate if one_b is radarstate.leadOne else _RadarStateProxy(one_b, radarstate.leadTwo)
    one = self._stop_gap_bias(one)
    if self._lead_smooth_enabled:
      one = self._smoother.update(one, self._stability.churn)  # de-jitter a churning lead (anti follow-hunt)
    if not self._vlead_damp_enabled:
      return _RadarStateProxy(one, two)                       # flicker-hold (A) only
    return _RadarStateProxy(self._one.smooth(one), self._two.smooth(two))
