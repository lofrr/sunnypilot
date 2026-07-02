"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance conditions the lead the longitudinal MPC follows on a noisy (TSS2-class) radar. It NEVER
reports a farther-or-faster lead than reality, so braking is always >= stock. Three mechanisms:
  * flicker-hold: keep a just-dropped, recently-sustained lead alive (dead-reckoned) through a brief radar
    dropout so the MPC does not lose and re-grab it (which reads as a phantom release then a catch-up brake);
  * churn smoother: a short SYMMETRIC EMA on a trackId-churning lead's dRel/vLead/vRel so the MPC stops
    hunting the gap (removes the follow-jitter that reads as rubber-banding);
  * stop-gap: near a (near-)stopped lead at low speed report dRel a touch closer so the MPC's own smooth stop
    settles farther back (the Prius TSS2 stock crawl creeps in to ~1.5 m). Monotone (closer => brake >= stock).
Also publishes a read-only lead-instability flag (telemetry). Disabled => byte-stock passthrough always.
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

LOW_SPEED_PASSTHROUGH_V = 5.0   # m/s: below this, no flicker-hold (holding a stale lead near a stop would
                                # delay the launch); the churn smoother still runs down to CREEP_PASSTHROUGH_V
CREEP_PASSTHROUGH_V = 1.0       # m/s: below this, full byte-stock passthrough (protect the stock stop distance)

SWITCH_DREL = 8.0              # m, dRel jump = a track switch (used by the instability detector)

# Lead-instability detector (telemetry only): flags a bimodal/bouncing radar lead.
STABILITY_WINDOW = 5           # frames (~0.25s @ 20Hz)
VLEAD_SPREAD = 4.0            # m/s, vLead range over the window above which the lead is "unstable"
ID_CHURN_WINDOW = 10          # frames (~0.5s) for radarTrackId-churn detection (steady lead, flipping ids)
ID_CHURN = 3                 # trackId switches in the window above which the lead is "unstable" (follow-hunting)

# Churn smoother: during trackId churn the per-track dRel/vRel jitter makes the MPC hunt the follow gap. A
# short SYMMETRIC EMA on the churning lead removes the jitter so the MPC sees a steady lead. Active ONLY
# during churn (NOT bimodal vLead -> never averages two real tracks). Bounded symmetric lag ~LEAD_SMOOTH_TAU.
LEAD_SMOOTH_TAU = 0.5          # s, EMA time constant
LEAD_SMOOTH_HOLD = 20          # frames (~1s): keep smoothing through brief churn gaps (churn toggles on/off)

# Stop-gap: near a (near-)stopped lead at low speed, report dRel up to STOP_GAP_M closer so the MPC's own
# smooth stop terminates that much farther back (stock Prius crawl-creeps to ~1.5 m). Monotone (only reports
# closer => brake >= stock). Ramps in below the regime dRel and out as the lead starts moving; releases on
# launch as ego speed rises past STOP_GAP_VEGO.
STOP_GAP_M = 2.5               # m: max dRel reduction = added standstill gap
STOP_GAP_VEGO = 8.0            # m/s: only below this ego speed
STOP_GAP_VLEAD = 1.5           # m/s: only behind a (near-)stopped lead; ramps out as vLead rises to this
STOP_GAP_REGIME_DREL = 12.0    # m: bias ramps in below this dRel
STOP_GAP_RAMP_BAND = 2.0       # m: ramp-in band (full offset below REGIME_DREL - RAMP_BAND)
STOP_GAP_MIN_DREL = 2.0        # m: never report a lead closer than this


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
    if not lead.status or v_ego < CREEP_PASSTHROUGH_V:
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
    # CP accepted for the planner's constructor signature; unused.
    self._params = params or Params()
    self._frame = 0
    self._v_ego = 0.0
    self._enabled = self._params.get_bool("RadarDistance")
    self._one = _LeadHold()
    self._two = _LeadHold()
    self._stability = _LeadStability()
    self._smoother = _LeadSmoother()

  def _read_params(self) -> None:
    enabled = self._params.get_bool("RadarDistance")
    if not enabled and self._enabled:
      self._one.reset()
      self._two.reset()
      self._smoother.reset()
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
    # Report a (near-)stopped lead up to STOP_GAP_M closer at low speed so the MPC's own smooth stop ends
    # that much farther back. Monotone (only ever reports closer). No-op outside the regime.
    if not lead.status or lead.vLead > STOP_GAP_VLEAD or self._v_ego > STOP_GAP_VEGO or lead.dRel <= STOP_GAP_MIN_DREL:
      return lead
    d_ramp = min(max((STOP_GAP_REGIME_DREL - lead.dRel) / STOP_GAP_RAMP_BAND, 0.0), 1.0)
    v_ramp = min(max((STOP_GAP_VLEAD - lead.vLead) / STOP_GAP_VLEAD, 0.0), 1.0)
    offset = STOP_GAP_M * d_ramp * v_ramp
    if offset < 0.05:
      return lead
    return _BiasedLead(lead, max(lead.dRel - offset, STOP_GAP_MIN_DREL))

  def smooth_radarstate(self, radarstate):
    self._stability.update(radarstate.leadOne, self._v_ego)   # telemetry, runs every cycle
    if not self._enabled:
      return radarstate                                       # off: byte-stock passthrough
    two = radarstate.leadTwo
    if self._v_ego >= LOW_SPEED_PASSTHROUGH_V:
      one = self._one.step(radarstate.leadOne)                # flicker-hold ...
      two = self._two.step(radarstate.leadTwo)
      one = self._smoother.update(one, self._stability.churn) # ... + churn de-jitter (anti follow-hunt)
    elif self._v_ego >= CREEP_PASSTHROUGH_V:
      # creep band: churn de-jitter ONLY (symmetric EMA), no flicker-hold (a stale held lead would delay launch)
      one = self._smoother.update(radarstate.leadOne, self._stability.churn)
    else:
      one = radarstate.leadOne                                # full standstill: no hold/smoothing
    one = self._stop_gap_bias(one)                            # low-speed near-stopped: settle farther back
    if one is radarstate.leadOne and two is radarstate.leadTwo:
      return radarstate                                       # nothing changed -> byte-stock object
    return _RadarStateProxy(one, two)
