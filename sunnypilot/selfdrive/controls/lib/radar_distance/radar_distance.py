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

# How many frames a _JumpGuard/_LeadSmoother reference can go without a step()/update() call (below the speed
# gate that call is skipped entirely) before it's discarded as stale rather than diffed/EMA'd against. Must be
# small next to a real stop (the bug this guards against ran 3200+ frames/160s) but comfortably bigger than an
# incidental single skipped cycle from v_ego dithering right at LOW_SPEED_PASSTHROUGH_V/CREEP_PASSTHROUGH_V --
# a flat ">1" here defeated the same-cycle farther-jump guard on every dithered cycle and let a real jitter
# glitch leak into the smoother's EMA for up to ~1s.
REFERENCE_STALE_FRAMES = 20    # ~1.0s

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
    self._last_frame = 0

  def reset(self):
    self._d = None
    self._vl = None
    self._vr = None
    self._hold = 0
    self._last_frame = 0

  def update(self, lead, noisy: bool, frame: int):
    # A held lead's dRel/vLead is a stale extrapolation -- feeding it into the EMA would both hide that from
    # downstream (wraps it into a _SmoothedLead) and pollute _d/_vl/_vr, lagging the real value's recovery.
    if isinstance(lead, _HeldLead):
      self.reset()
      return lead
    # update() is only called above CREEP_PASSTHROUGH_V (see smooth_radarstate), so _hold and the EMA state
    # (_d/_vl/_vr) freeze for the entire duration of any full standstill. Resuming and EMA-ing a real, opening
    # lead against that frozen _d as if no time had passed lags dRel toward the stale, closer pre-stop value
    # for up to LEAD_SMOOTH_TAU-ish seconds -- same bug class as _JumpGuard/_LeadHold's frame-based staleness
    # fixes. Treat a gap since the last call larger than REFERENCE_STALE_FRAMES (not a flat 1) as no state at
    # all, so an incidental single skipped cycle from v_ego dithering right at the gate doesn't itself wipe the
    # EMA's short-term jitter-suppression memory.
    if self._last_frame and frame - self._last_frame > REFERENCE_STALE_FRAMES:
      self.reset()
    self._last_frame = frame
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
    self._last_frame = 0
    self._hold = 0
    self._grace_used = False

  def reset(self):
    self._last = None
    self._last_frame = 0
    self._hold = 0
    self._grace_used = False

  def step(self, raw, frame):
    if not raw.status:
      self.reset()
      return raw

    # _last is only a valid reference to diff against if it's recent. smooth_radarstate() stops calling step()
    # below LOW_SPEED_PASSTHROUGH_V (see _LeadHold), so after any low-speed gap (a full stop, a slow zone)
    # _last can be arbitrarily many real seconds old. Diffing a fresh reading against a stale one as if it were
    # a same-cycle transient rejects a real, large, entirely legitimate change (e.g. a lead that pulled away
    # during the gap) as a fusion glitch and holds a phantom lead -- measured causing a hard, unwarranted brake
    # on a real route (launch from a stop after the lead had long since moved on). Treat a stale _last as no
    # reference at all: pass raw through and re-baseline. REFERENCE_STALE_FRAMES (not a flat 1) so an
    # incidental single skipped cycle from v_ego dithering right at the gate doesn't itself defeat the guard.
    stale = self._last is not None and (frame - self._last_frame) > REFERENCE_STALE_FRAMES

    if not stale and self._last is not None and (raw.dRel - self._last[0]) > SWITCH_DREL and self._hold < JUMP_GUARD_MAX_HOLD:
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      self._hold += 1
      held_dRel = max(MIN_HELD_DREL, dRel0 - max(-vRel0, 0.0) * DT_MDL)
      self._last = (held_dRel, vRel0, vLead0, aLeadK0, aLeadTau0, prob0)
      self._last_frame = frame
      return _HeldLead(held_dRel, vRel0, vLead0, aLeadK0, aLeadTau0, min(prob0, FCW_PROB_CAP))

    # Hold cap reached on a lead that was closing: self-healing straight onto raw here would adopt a farther
    # reading than the trajectory already tracked, i.e. report a farther lead than reality for at least one
    # more cycle. Take exactly one bounded extra cycle at the last-held value first -- never a second, so this
    # can't turn into an indefinite hold on a lead that genuinely departed.
    if (not stale and self._hold >= JUMP_GUARD_MAX_HOLD and not self._grace_used and self._last is not None and
        self._last[1] < 0.0 and (raw.dRel - self._last[0]) > SWITCH_DREL):
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      self._grace_used = True
      self._last_frame = frame
      return _HeldLead(dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, min(prob0, FCW_PROB_CAP))

    self._hold = 0
    self._grace_used = False
    self._last = (raw.dRel, raw.vRel, raw.vLead, raw.aLeadK, raw.aLeadTau, raw.modelProb)
    self._last_frame = frame
    return raw


class _LeadHold:
  # step() takes the caller's absolute frame counter rather than counting its own calls: below
  # LOW_SPEED_PASSTHROUGH_V the caller stops calling step() at all (see smooth_radarstate), and a
  # self-incrementing counter would then stay frozen at whatever it was for however long that lasts -- on
  # resume it would read as "just a few frames since the last real sighting" no matter how much real time
  # (a full stop, a slow zone) actually passed, and could hand HOLD_MAX_FRAMES worth of stale credit to a
  # sighting from arbitrarily long ago. Comparing against the caller's frame counter makes the elapsed-frames
  # check correct regardless of how many cycles were skipped in between.
  def __init__(self):
    self._last = None
    self._sustained = 0
    self._real_frame = 0
    self._armed = False
    self._held_dRel = 0.0
    self._holding = False   # true once this hold episode has been reseeded from a real reading

  def reset(self):
    self.__init__()

  def step(self, raw, frame):
    if raw.status and raw.dRel > DROPOUT_DREL:
      self._last = (raw.dRel, raw.vRel, raw.vLead, raw.aLeadK, raw.aLeadTau, raw.modelProb)
      self._sustained += 1
      if self._sustained >= SUSTAIN_FRAMES:
        self._real_frame = frame
        self._armed = True
      self._holding = False   # back on a real sighting -- the next dropout starts a fresh hold episode
      return raw

    self._sustained = 0
    since_real = frame - self._real_frame
    if self._armed and self._last is not None and since_real <= HOLD_MAX_FRAMES:
      dRel0, vRel0, vLead0, aLeadK0, aLeadTau0, prob0 = self._last
      # Reseed _held_dRel from the real last-known value exactly once per hold episode, on whichever call
      # first starts holding -- NOT on since_real==1: since_real is elapsed REAL frames (see class docstring),
      # so any low-speed gap in between (step() skipped) makes since_real > 1 on the very first dropout call
      # actually made, and comparing it to 1 would silently skip the reseed -- leaving _held_dRel at its
      # stale/init value (0.0), which the next line's floor then clamps to MIN_HELD_DREL: a fabricated
      # near-bumper phantom lead, not a dead-reckoned extrapolation of the real one.
      if not self._holding:
        self._held_dRel = dRel0
        self._holding = True
      self._held_dRel = max(MIN_HELD_DREL, self._held_dRel - max(-vRel0, 0.0) * DT_MDL)
      return _HeldLead(self._held_dRel, vRel0, vLead0, min(aLeadK0, 0.0), aLeadTau0, min(prob0, FCW_PROB_CAP))

    self._armed = False
    self._holding = False
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
    self._frame += 1                                           # step()'s elapsed-frames basis; see _LeadHold
    self._stability.update(radarstate.leadOne, self._v_ego)   # telemetry, runs every cycle
    if not self._enabled:
      return radarstate                                       # off: byte-stock passthrough
    two = radarstate.leadTwo
    noisy = self._stability.churn or self._stability.same_track_noise
    if self._v_ego >= LOW_SPEED_PASSTHROUGH_V:
      one = self._jump_guard.step(radarstate.leadOne, self._frame)  # reject a same-cycle farther-jump transient ...
      one = self._one.step(one, self._frame)                  # ... + flicker-hold ...
      two = self._two.step(radarstate.leadTwo, self._frame)
      one = self._smoother.update(one, noisy, self._frame)     # ... + same-object de-jitter (anti follow-hunt)
    elif self._v_ego >= CREEP_PASSTHROUGH_V:
      # creep band: de-jitter ONLY (symmetric EMA), no flicker-hold (a stale held lead would delay launch)
      one = self._smoother.update(radarstate.leadOne, noisy, self._frame)
    else:
      one = radarstate.leadOne                                # full standstill: no hold/smoothing
    one = self._stop_gap_bias(one)                            # low-speed near-stopped: settle farther back
    if one is radarstate.leadOne and two is radarstate.leadTwo:
      return radarstate                                       # nothing changed -> byte-stock object
    return _RadarStateProxy(one, two)
