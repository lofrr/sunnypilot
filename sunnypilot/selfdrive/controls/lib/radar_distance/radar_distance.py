"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance conditions the lead the longitudinal MPC follows on a noisy (TSS2-class) radar. It NEVER
reports a farther-or-faster lead than reality, so braking is always >= stock. Four mechanisms:
  * jump-guard: reject a same-cycle FARTHER dRel jump on a lead that never dropped status (a vision/radar
    fusion transient during lead acquisition -- e.g. a cut-in whose vision distance estimate briefly
    disagrees with a solid radar track) by holding the last-trusted, closer reading instead of snapping back
    out. A closer jump of any size always passes immediately -- this only ever delays relief, never a brake;
  * flicker-hold: keep a just-dropped, recently-sustained lead alive (dead-reckoned) through a brief radar
    dropout so the MPC does not lose and re-grab it (which reads as a phantom release then a catch-up brake);
  * churn/noise smoother: a short EMA on a lead's dRel/vLead/vRel so the MPC stops hunting the gap (removes
    the follow-jitter that reads as rubber-banding and, on the sensor side, as a lead-detection "lurch").
    Covers two DISTINCT same-physical-object noise signatures: trackId churn (id flips frame-to-frame but the
    kinematics stay coherent -- one real lead getting re-labeled) and same-track noise (id stays constant but
    vLead itself is bimodal/bouncing -- one real lead with a noisy fusion/Doppler velocity read). Both are
    safe to EMA because the id evidence pins them to a SINGLE physical object; a bimodal vLead WITH the id
    also changing is left alone (ambiguous -- could be two really-different real objects) so this can never
    average two real tracks together. dRel is asymmetric -- closer accepted immediately, only farther is
    EMA-lagged -- so it can't hold a steadily-closing lead farther-than-true; vLead/vRel stay symmetric;
  * stop-gap: near a (near-)stopped lead at low speed report dRel a touch closer so the MPC's own smooth stop
    settles farther back (the Prius TSS2 stock crawl creeps in to ~1.5 m). Monotone (closer => brake >= stock).
    Overridden off by sustained lead motion (even slow creep) so it can't suppress a real, growing gap during
    a launch. Never runs on a held (jump-guard or flicker-hold) lead, since a hold's vLead/dRel are stale;
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

SWITCH_DREL = 4.0               # m, dRel jump = a track switch (used by the instability detector + jump-guard).

# Jump-guard: a same-cycle dRel jump this far FARTHER, on a lead that never dropped status, is treated as a
# fusion transient (not a real sudden separation) and held at the last-trusted value for a bounded number of
# frames. Self-heals fast so a genuinely-departing lead is never held stale for long.
JUMP_GUARD_MAX_HOLD = 10       # frames (~0.5s)

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

# Stop-gap creep override: a lead creeping forward slowly can sit under STOP_GAP_VLEAD for many seconds
# without crossing it, so the bias keeps suppressing a real, growing gap. Sustained motion this long overrides
# the bias off regardless of how slow. The counter decays (not just holds) on a sub-threshold frame, so it
# takes sustained motion, not noise straddling the threshold, to reach the cap.
STOP_GAP_CREEP_V = 0.03        # m/s: a truly-stopped lead reads exactly 0.0; treat anything above this as motion
STOP_GAP_CREEP_HOLD_S = 1.5    # s: this much sustained motion overrides the bias off
STOP_GAP_CREEP_HOLD_FRAMES = int(STOP_GAP_CREEP_HOLD_S / DT_MDL)


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
  # EMA on a noisy same-physical-object lead's dRel/vLead/vRel (jitter removal; see _LeadStability for what
  # qualifies as "same object"). A hold keeps it active through brief noise gaps (the trigger toggles on/off);
  # passthrough + reset only after the hold lapses. dRel is ASYMMETRIC: a closer raw reading is accepted
  # immediately (never delay awareness of closer -- the file's own invariant), only a FARTHER raw reading is
  # EMA-lagged (reject noise in that direction). Without this, a lead that's genuinely closing steadily while
  # noisy (even briefly) gets held farther-than-true for the full LEAD_SMOOTH_HOLD window, then snaps -- a
  # false-relief-then-correction that itself becomes a hard brake.
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

  def update(self, lead, noisy: bool):
    # A held lead's dRel/vLead is a stale extrapolation -- feeding it into the EMA would both hide that from
    # downstream (wraps it into a _SmoothedLead) and pollute _d/_vl/_vr, lagging the real value's recovery.
    if isinstance(lead, _HeldLead):
      self.reset()
      return lead
    self._hold = LEAD_SMOOTH_HOLD if noisy else self._hold - 1
    if self._hold <= 0 or not lead.status:
      self.reset()
      return lead
    if self._d is None:
      self._d, self._vl, self._vr = lead.dRel, lead.vLead, lead.vRel
      return lead
    a = DT_MDL / LEAD_SMOOTH_TAU
    self._d = lead.dRel if lead.dRel < self._d else self._d + (lead.dRel - self._d) * a
    self._vl += (lead.vLead - self._vl) * a
    self._vr += (lead.vRel - self._vr) * a
    return _SmoothedLead(lead, self._d, self._vl, self._vr)


class _JumpGuard:
  # Rejects a same-cycle FARTHER dRel jump on a lead that never dropped status (a vision/radar fusion
  # transient, e.g. a cut-in whose vision distance estimate briefly disagrees with a solid radar track before
  # the match locks on) by holding the last-trusted reading, extrapolated by its own vRel, for a bounded
  # number of frames. A CLOSER jump of any size always passes through immediately -- this can only ever delay
  # relief, never a brake -- and it self-heals after JUMP_GUARD_MAX_HOLD frames if the jump was real.
  # modelProb is capped at FCW_PROB_CAP on the held lead (same as _LeadHold's flicker-hold, below) -- a held
  # reading is no longer confirmed fresh, so it must not carry enough confidence to trip the stock FCW gate.
  def __init__(self):
    self._last = None
    self._hold = 0

  def reset(self):
    self._last = None
    self._hold = 0

  def step(self, raw):
    if not raw.status:
      self.reset()
      return raw

    if self._last is not None and (raw.dRel - self._last[0]) > SWITCH_DREL and self._hold < JUMP_GUARD_MAX_HOLD:
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      self._hold += 1
      held_dRel = max(MIN_HELD_DREL, dRel0 - max(-vRel0, 0.0) * DT_MDL)
      self._last = (held_dRel, vRel0, vLead0, aLeadK0, aLeadTau0, prob0)
      return _HeldLead(held_dRel, vRel0, vLead0, aLeadK0, aLeadTau0, min(prob0, FCW_PROB_CAP))

    self._hold = 0
    self._last = (raw.dRel, raw.vRel, raw.vLead, raw.aLeadK, raw.aLeadTau, raw.modelProb)
    return raw


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
  # Also derives same_track_noise: a bimodal/bouncing vLead while radarTrackId sat CONSTANT the whole window
  # -- i.e. the id evidence pins the noise to one physical object (a Doppler/fusion-noisy velocity read on one
  # real lead), so it is safe to feed the smoother (see _LeadSmoother). A bimodal vLead WITH the id also
  # changing stays outside same_track_noise (could be two really-different real objects at different speeds)
  # and is left unmitigated, same as before. dRel track-jumps are deliberately excluded here: while status
  # stays True (this class's own precondition), a repeated FARTHER dRel jump is already absorbed by
  # _JumpGuard upstream (same SWITCH_DREL threshold), so adding it here would just double up on the same
  # signal rather than covering a real gap.
  def __init__(self):
    self._v = deque(maxlen=STABILITY_WINDOW)
    self._d = deque(maxlen=STABILITY_WINDOW)
    self._id = deque(maxlen=ID_CHURN_WINDOW)
    self.unstable = False
    self.churn = False
    self.same_track_noise = False

  def reset(self):
    self._v.clear()
    self._d.clear()
    self._id.clear()
    self.unstable = False
    self.churn = False
    self.same_track_noise = False

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
    recent_ids = ids[-STABILITY_WINDOW:]
    same_track = recent_ids[0] > 0 and len(set(recent_ids)) == 1
    self.churn = id_churn >= ID_CHURN and v_spread <= VLEAD_SPREAD   # steady lead, flipping ids (not bimodal)
    self.same_track_noise = same_track and v_spread > VLEAD_SPREAD
    self.unstable = v_spread > VLEAD_SPREAD or d_jumps >= 2 or self.churn


class RadarDistanceController:
  def __init__(self, CP: structs.CarParams, params=None):
    # CP accepted for the planner's constructor signature; unused.
    self._params = params or Params()
    self._frame = 0
    self._v_ego = 0.0
    self._enabled = self._params.get_bool("RadarDistance")
    self._jump_guard = _JumpGuard()
    self._one = _LeadHold()
    self._two = _LeadHold()
    self._stability = _LeadStability()
    self._smoother = _LeadSmoother()
    self._creep_frames = 0
    self._creep_released = False

  def _read_params(self) -> None:
    enabled = self._params.get_bool("RadarDistance")
    if not enabled and self._enabled:
      self._jump_guard.reset()
      self._one.reset()
      self._two.reset()
      self._smoother.reset()
      self._creep_frames = 0
      self._creep_released = False
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
    in_regime = (lead.status and lead.vLead <= STOP_GAP_VLEAD and
                 self._v_ego <= STOP_GAP_VEGO and lead.dRel > STOP_GAP_MIN_DREL)
    if not in_regime:
      self._creep_frames = 0
      self._creep_released = False
      return lead

    # A held (stale) lead's near-zero vLead can still satisfy the regime check on a lead that's departed --
    # skip biasing it, but freeze the creep latch rather than resetting it (an unrelated jump-guard glitch
    # shouldn't undo motion the lead already earned).
    if isinstance(lead, _HeldLead):
      return lead

    if lead.vLead > STOP_GAP_CREEP_V:
      self._creep_frames = min(self._creep_frames + 1, STOP_GAP_CREEP_HOLD_FRAMES)
      if self._creep_frames >= STOP_GAP_CREEP_HOLD_FRAMES:
        self._creep_released = True
    else:
      # decay: a sub-threshold frame undoes one frame of "motion" credit, so only SUSTAINED motion (not
      # cumulative noise straddling the threshold) can reach the cap.
      self._creep_frames = max(self._creep_frames - 1, 0)
    if self._creep_released:                          # once latched: sticky -- a single-frame near-zero blip
      return lead                                      # afterward (e.g. sensor noise mid-creep) can't re-suppress it

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
    noisy = self._stability.churn or self._stability.same_track_noise
    if self._v_ego >= LOW_SPEED_PASSTHROUGH_V:
      one = self._jump_guard.step(radarstate.leadOne)         # reject a same-cycle farther-jump transient ...
      one = self._one.step(one)                               # ... + flicker-hold ...
      two = self._two.step(radarstate.leadTwo)
      one = self._smoother.update(one, noisy)                 # ... + same-object de-jitter (anti follow-hunt)
    elif self._v_ego >= CREEP_PASSTHROUGH_V:
      # creep band: de-jitter ONLY (symmetric EMA), no flicker-hold (a stale held lead would delay launch)
      one = self._smoother.update(radarstate.leadOne, noisy)
    else:
      one = radarstate.leadOne                                # full standstill: no hold/smoothing
    one = self._stop_gap_bias(one)                            # low-speed near-stopped: settle farther back
    if one is radarstate.leadOne and two is radarstate.leadTwo:
      return radarstate                                       # nothing changed -> byte-stock object
    return _RadarStateProxy(one, two)
