"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

RadarDistance is a pure lead DE-NOISER: flicker-hold + churn smoother + instability telemetry, and nothing
else (no dRel biasing). These tests pin: off / low-speed == byte-stock (stock stop distance); the hold is
obstacle-monotone (brake >= stock) and bounded; the churn smoother de-jitters only a track-flipping lead;
and the instability flag is telemetry that runs regardless of the gate.
"""

from types import SimpleNamespace

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.radar_distance.radar_distance import \
  RadarDistanceController, HOLD_MAX_FRAMES, FCW_PROB_CAP, LOW_SPEED_PASSTHROUGH_V, CREEP_PASSTHROUGH_V, \
  DROPOUT_DREL, STOP_GAP_MIN_DREL, STOP_GAP_VEGO, STOP_GAP_VLEAD, STOP_GAP_REGIME_DREL, SWITCH_DREL, \
  JUMP_GUARD_MAX_HOLD, STOP_GAP_CREEP_V, STOP_GAP_CREEP_HOLD_FRAMES

COMFORT_BRAKE = 2.5


class FakeParams:
  def __init__(self, store=None):
    self.store = dict(store or {})

  def get_bool(self, key):
    return bool(self.store.get(key, False))


def lead(status=True, dRel=40.0, vRel=-2.0, vLead=18.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.95, radarTrackId=-1):
  return SimpleNamespace(status=status, dRel=dRel, yRel=0.0, vRel=vRel, vLead=vLead, vLeadK=vLead,
                         aLeadK=aLeadK, aLeadTau=aLeadTau, modelProb=modelProb, radarTrackId=radarTrackId)


def rs(one, two=None):
  return SimpleNamespace(leadOne=one, leadTwo=two or lead(status=False, dRel=0.0, modelProb=0.0))


def obstacle(ld):
  return ld.dRel + ld.vLead ** 2 / (2 * COMFORT_BRAKE)


def ctrl(enabled=True, v_ego=10.0):
  c = RadarDistanceController(CP=SimpleNamespace(), params=FakeParams({'RadarDistance': enabled}))
  c._v_ego = v_ego   # above the low-speed gate so the hold + smoother run
  return c


def churn_frames(n, d_a=40.0, d_b=42.0, vLead=18.0):
  # a steady lead whose radarTrackId flips every frame (dRel jitters with it) -> the churn detector fires and
  # the smoother should de-jitter dRel. vLead is steady so it is NOT flagged bimodal (never averages 2 tracks).
  for i in range(n):
    even = i % 2 == 0
    yield lead(dRel=d_a if even else d_b, vLead=vLead, vRel=-1.0, radarTrackId=1 if even else 2)


# --- off / low-speed == byte-stock ------------------------------------------------------------------------

def test_disabled_is_identity():
  c = ctrl(enabled=False)
  r = rs(lead())
  assert c.smooth_radarstate(r) is r                 # byte-stock passthrough


def test_valid_lead_passthrough():
  c = ctrl()
  one = lead(dRel=40.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one                          # clean lead, no churn -> unchanged


def test_full_standstill_outside_stopgap_is_passthrough():
  # Full standstill (< CREEP_PASSTHROUGH_V), lead OUTSIDE the stop-gap regime (far): no hold, no smoothing,
  # no bias -> the EXACT raw radarstate object (byte-identical). The stop-gap only engages inside its regime.
  c = ctrl(v_ego=CREEP_PASSTHROUGH_V - 0.5)
  r = rs(lead(dRel=STOP_GAP_REGIME_DREL + 8.0, vLead=0.5))
  assert c.smooth_radarstate(r) is r


def test_creep_dejitters_churn_but_no_hold():
  # Creep band [CREEP, LOW_SPEED): the churn smoother runs (de-jitter -> smooth stop-and-go), but the
  # flicker-hold does NOT (a dropped/departed lead must not be held, or launch would be delayed).
  # vLead>STOP_GAP_VLEAD so the stop-gap stays out and this isolates the EMA.
  c = ctrl(v_ego=(CREEP_PASSTHROUGH_V + LOW_SPEED_PASSTHROUGH_V) / 2)
  out = None
  for f in churn_frames(30, d_a=6.0, d_b=8.0, vLead=3.0):
    out = c.smooth_radarstate(rs(f))
  assert 6.0 < out.leadOne.dRel < 8.0                # jitter smoothed
  # a dropout in the creep band is NOT held -> raw passes through (no stale lead)
  drop = rs(lead(status=False, dRel=0.0, modelProb=0.0))
  assert c.smooth_radarstate(drop) is drop


def test_creep_clean_lead_passthrough():
  # creep band, steady moving lead (no churn, outside stop-gap regime) -> exact raw object (unbiased)
  c = ctrl(v_ego=(CREEP_PASSTHROUGH_V + LOW_SPEED_PASSTHROUGH_V) / 2)
  r = rs(lead(dRel=4.0, vLead=2.5, radarTrackId=3))
  assert c.smooth_radarstate(r) is r


# --- stop-gap (settle farther back from a near-stopped lead) ----------------------------------------------

def test_stop_gap_pulls_stopped_lead_closer():
  c = ctrl(v_ego=2.0)
  one = lead(dRel=6.0, vLead=0.0, vRel=-1.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne.dRel < 6.0                      # reported closer -> MPC stops farther back
  assert obstacle(out.leadOne) <= obstacle(one) + 1e-6   # brake >= stock (obstacle never farther)


def test_stop_gap_monotone_never_farther():
  c = ctrl(v_ego=3.0)
  for d in (4.0, 6.0, 9.0, 11.0):
    out = c.smooth_radarstate(rs(lead(dRel=d, vLead=0.0)))
    assert out.leadOne.dRel <= d + 1e-6


def test_stop_gap_min_floor():
  c = ctrl(v_ego=2.0)
  out = c.smooth_radarstate(rs(lead(dRel=STOP_GAP_MIN_DREL + 0.5, vLead=0.0)))
  assert out.leadOne.dRel >= STOP_GAP_MIN_DREL - 1e-6


def test_stop_gap_off_when_disabled():
  c = ctrl(enabled=False, v_ego=2.0)
  r = rs(lead(dRel=6.0, vLead=0.0))
  assert c.smooth_radarstate(r) is r                 # disabled -> stock stop distance


def test_stop_gap_moving_lead_no_change():
  c = ctrl(v_ego=2.0)
  out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=STOP_GAP_VLEAD + 1.0)))
  assert out.leadOne.dRel == pytest.approx(6.0)      # lead moving -> not a stop


def test_stop_gap_high_speed_no_change():
  c = ctrl(v_ego=STOP_GAP_VEGO + 2.0)
  out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=0.0)))
  assert out.leadOne.dRel == pytest.approx(6.0)      # above the stop regime -> unbiased


def test_stop_gap_far_lead_no_change():
  c = ctrl(v_ego=2.0)
  d = STOP_GAP_REGIME_DREL + 5.0
  out = c.smooth_radarstate(rs(lead(dRel=d, vLead=0.0)))
  assert out.leadOne.dRel == pytest.approx(d)        # beyond the ramp-in regime -> unbiased


# --- stop-gap creep override (sustained lead motion releases the bias even below STOP_GAP_VLEAD) ----------

def test_stop_gap_creep_releases_after_sustained_motion():
  # route 550a71ee4c7a7fbe/000004a4--c9c4691959, t~1126-1138: a lead crept forward at 0.3-0.6 m/s (well under
  # STOP_GAP_VLEAD) for 9+ seconds; without the override the bias suppressed the whole real gap growth the
  # entire time, producing a 9+ second launch delay. Real dRel grows steadily 4.0 -> 4.5m over this window.
  c = ctrl(v_ego=0.0)
  d = 4.0
  out = None
  for i in range(STOP_GAP_CREEP_HOLD_FRAMES + 5):
    d += 0.05
    out = c.smooth_radarstate(rs(lead(dRel=d, vLead=0.4)))
    if i < STOP_GAP_CREEP_HOLD_FRAMES - 1:
      assert out.leadOne.dRel < d - 1e-6                # still suppressed while creep hasn't sustained yet
  assert out.leadOne.dRel == pytest.approx(d)            # released after STOP_GAP_CREEP_HOLD_FRAMES of motion


def test_stop_gap_creep_release_is_sticky_through_a_momentary_zero_blip():
  # A single-frame return to exactly 0.0 (real sensor behavior mid-creep, not just noise) must not re-arm the
  # bias once sustained creep has already released it -- this was the actual bug: the real route's lead
  # dipped to vLead=0.00 for one frame mid-launch and the bias briefly re-suppressed the gap right as a result.
  c = ctrl(v_ego=0.0)
  d = 4.0
  for _ in range(STOP_GAP_CREEP_HOLD_FRAMES):
    d += 0.05
    c.smooth_radarstate(rs(lead(dRel=d, vLead=0.4)))
  d += 0.05
  out = c.smooth_radarstate(rs(lead(dRel=d, vLead=0.0)))  # exact-zero blip
  assert out.leadOne.dRel == pytest.approx(d)             # still released, not re-suppressed
  d += 0.05
  out = c.smooth_radarstate(rs(lead(dRel=d, vLead=0.3)))  # creep resumes
  assert out.leadOne.dRel == pytest.approx(d)


def test_stop_gap_creep_counter_resets_on_a_genuine_new_stop():
  # Leaving the bias regime entirely (lead departs, or ego speeds past STOP_GAP_VEGO) must re-arm the
  # override so a LATER, unrelated near-stop encounter isn't permanently exempted by a stale latch.
  c = ctrl(v_ego=0.0)
  d = 4.0
  for _ in range(STOP_GAP_CREEP_HOLD_FRAMES):
    d += 0.05
    c.smooth_radarstate(rs(lead(dRel=d, vLead=0.4)))
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))   # lead lost -> regime exit
  out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=0.0)))               # fresh near-stop encounter
  assert out.leadOne.dRel < 6.0                                          # bias re-armed, active again


def test_stop_gap_creep_below_threshold_never_releases():
  c = ctrl(v_ego=0.0)
  out = None
  for _ in range(STOP_GAP_CREEP_HOLD_FRAMES + 10):
    out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=STOP_GAP_CREEP_V * 0.5)))
  assert out.leadOne.dRel < 6.0 - 1e-6                # vLead never exceeds the creep threshold -> stays biased


def test_stop_gap_creep_intermittent_noise_never_releases():
  # route 550a71ee4c7a7fbe/000004b6--d4a8ac3352, t~678-690s: a genuinely-stopped lead's vLead noise blipped
  # above STOP_GAP_CREEP_V on roughly half the frames (never below zero motion overall, never sustained) --
  # the old monotonic-only counter still accumulated those blips to the cap over enough frames and falsely
  # latched the bias off mid-stop, producing a same-cycle jump in the reported gap. Alternating strictly
  # above/below the threshold every frame, for far longer than STOP_GAP_CREEP_HOLD_FRAMES, must never release.
  c = ctrl(v_ego=0.0)
  out = None
  for i in range(STOP_GAP_CREEP_HOLD_FRAMES * 4):
    vLead = STOP_GAP_CREEP_V + 0.05 if i % 2 == 0 else 0.0
    out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=vLead)))
  assert out.leadOne.dRel < 6.0 - 1e-6                # never sustained -> never releases, even after 4x the hold


def test_stop_gap_creep_sustained_after_intermittent_noise_still_releases():
  # the decay must not make the override permanently harder to reach -- real sustained motion right after a
  # noisy patch still releases within the normal HOLD window (decay only undoes noise, doesn't add a penalty).
  c = ctrl(v_ego=0.0)
  for i in range(STOP_GAP_CREEP_HOLD_FRAMES):
    vLead = STOP_GAP_CREEP_V + 0.05 if i % 2 == 0 else 0.0
    c.smooth_radarstate(rs(lead(dRel=6.0, vLead=vLead)))
  out = None
  for _ in range(STOP_GAP_CREEP_HOLD_FRAMES + 5):
    out = c.smooth_radarstate(rs(lead(dRel=6.0, vLead=STOP_GAP_CREEP_V + 0.05)))
  assert out.leadOne.dRel == pytest.approx(6.0)       # sustained motion still releases in bounded time


def test_stop_gap_does_not_double_bias_a_jump_guard_held_lead():
  # route 550a71ee4c7a7fbe/000004bc--d9e0efd5ac, t~1563.4-1563.9: a lead sitting near-stopped (vLead~0,
  # inside the stop-gap regime) departs fast enough that a single raw dRel jump exceeds jump-guard's
  # SWITCH_DREL. Jump-guard holds its OLD, stale (near-zero vLead0) reading rather than the new, fast, real
  # one -- and that held vLead0 STILL satisfies the stop-gap regime check (it looks near-stopped), so without
  # the skip, stop-gap piles a SECOND closer-bias on top of an already-stale value. On the real route this
  # compounding dropped the reported gap far enough (raw ~16m -> ~2-5m) to fool the MPC's own forward-solve
  # into a spurious FCW during a real launch. Fixed: stop-gap must skip a held (stale) lead entirely.
  v_ego = (LOW_SPEED_PASSTHROUGH_V + STOP_GAP_VEGO) / 2   # in-band for jump-guard hold AND stop-gap regime
  c = ctrl(v_ego=v_ego)
  dRel0, vRel0, vLead0 = 7.5, -0.1, 0.2                  # near-stopped baseline, trusted
  c.smooth_radarstate(rs(lead(dRel=dRel0, vRel=vRel0, vLead=vLead0)))
  jumped = c.smooth_radarstate(rs(lead(dRel=dRel0 + SWITCH_DREL + 1.0, vRel=6.5, vLead=6.8)))  # real, fast departure
  expected_held = dRel0 - max(-vRel0, 0.0) * DT_MDL      # jump-guard's own extrapolation -- no further bias
  assert jumped.leadOne.dRel == pytest.approx(expected_held, abs=1e-6)


def test_smoother_does_not_launder_a_jump_guard_hold_during_churn():
  # A churn episode (real radarTrackId flapping, steady kinematics) actively engaging the smoother right as a
  # jump-guard hold begins must not let the smoother wrap the held lead into a _SmoothedLead -- that would hide
  # "this is stale" from stop-gap's held-lead check and let a second closer-bias stack on top.
  v_ego = (LOW_SPEED_PASSTHROUGH_V + STOP_GAP_VEGO) / 2   # in-band for jump-guard hold AND stop-gap regime
  c = ctrl(v_ego=v_ego)
  d = 9.0
  for i in range(8):
    tid = 1 if i % 2 == 0 else 2
    c.smooth_radarstate(rs(lead(dRel=d, vRel=-6.3, vLead=0.2, radarTrackId=tid)))
    dRel0 = d
    d -= 0.15
  assert c.lead_unstable()                               # churn primed the smoother
  vRel0 = -6.3
  jumped = c.smooth_radarstate(rs(lead(dRel=dRel0 + SWITCH_DREL + 1.0, vRel=6.5, vLead=6.8, radarTrackId=1)))
  expected_held = dRel0 - max(-vRel0, 0.0) * DT_MDL
  assert type(jumped.leadOne).__name__ == '_HeldLead'
  assert jumped.leadOne.dRel == pytest.approx(expected_held, abs=1e-6)


def test_stop_gap_creep_latch_survives_an_unrelated_jump_guard_hold():
  # A creep-release already earned (sustained real motion) must not be wiped by an unrelated jump-guard hold
  # that happens to land on the same lead -- the hold is a one-cycle fusion transient, not evidence the lead
  # stopped moving again.
  v_ego = (LOW_SPEED_PASSTHROUGH_V + STOP_GAP_VEGO) / 2   # in-band for jump-guard hold AND stop-gap regime
  c = ctrl(v_ego=v_ego)
  d = 4.0
  for _ in range(STOP_GAP_CREEP_HOLD_FRAMES + 2):
    d += 0.05
    c.smooth_radarstate(rs(lead(dRel=d, vRel=-6.0, vLead=0.4)))
  assert c._creep_released
  dRel0 = d
  held = c.smooth_radarstate(rs(lead(dRel=dRel0 + SWITCH_DREL + 1.0, vRel=6.5, vLead=6.8)))
  assert type(held.leadOne).__name__ == '_HeldLead'       # bias correctly skipped on the held cycle
  assert c._creep_released                                # but the earned latch must survive the glitch
  d += 0.05
  out = c.smooth_radarstate(rs(lead(dRel=d, vRel=-0.4, vLead=0.4)))
  assert out.leadOne.dRel == pytest.approx(d)              # creep resumes unbiased, no re-suppression


def test_low_speed_override_lead_passthrough():
  # radard low_speed_override emits a real closest-track lead with modelProb=0.0. It must be honored, not
  # rejected in favor of a stale farther held lead (which would under-brake / stop too close).
  c = ctrl()
  one = lead(status=True, dRel=2.5, vRel=0.0, vLead=0.0, modelProb=0.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one


# --- jump-guard (reject a same-cycle farther fusion transient) --------------------------------------------

def test_jump_guard_holds_farther_transient():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=27.92, vRel=-5.60, vLead=24.47, radarTrackId=1058)))
  out = c.smooth_radarstate(rs(lead(dRel=38.88, vRel=-3.19, vLead=26.91, radarTrackId=-1)))
  assert out.leadOne.dRel < 30.0                     # farther jump rejected, held near the trusted value
  assert out.leadOne.status is True


def test_jump_guard_passes_closer_jump_immediately():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-2.0, vLead=18.0)))
  out = c.smooth_radarstate(rs(lead(dRel=27.0, vRel=-5.0, vLead=15.0)))   # big CLOSER jump
  assert out.leadOne.dRel == pytest.approx(27.0)     # closer always passes through -- never delays a brake


def test_jump_guard_replays_real_route_whiplash():
  # route 550a71ee4c7a7fbe/00000498--0704864d6a, t~402.2-402.8: a merging lead's vision distance estimate
  # whiplashed 27.92 -> 38.88 -> 37.69 -> 37.20 -> 26.84 for ~0.3s while a solid radar track sat at ~27m the
  # whole time. The guard should smooth the farther excursion into a monotone converge toward the real value.
  c = ctrl()
  raw = [
    (74.18, -4.05, 25.77, -1), (53.21, -3.55, 26.33, -1), (47.42, -3.23, 26.67, -1),
    (42.64, -3.50, 26.42, -1), (43.22, -3.49, 26.49, -1), (40.03, -3.04, 26.96, -1),
    (39.50, -3.29, 26.74, -1), (27.92, -5.60, 24.47, 1058), (38.88, -3.19, 26.91, -1),
    (37.69, -3.09, 27.04, -1), (37.20, -2.77, 27.39, -1), (26.84, -5.80, 24.37, 1058),
  ]
  out = None
  for dRel, vRel, vLead, tid in raw:
    out = c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=vLead, radarTrackId=tid)))
  assert out.leadOne.dRel == pytest.approx(26.84)    # real value recovered exactly once raw resumes reporting it
  # peak reported dRel during the excursion never revisits the raw 38.88 spike
  seen = []
  c = ctrl()
  for dRel, vRel, vLead, tid in raw:
    seen.append(c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=vLead, radarTrackId=tid))).leadOne.dRel)
  assert max(seen[8:11]) < 30.0                      # the 3 farther-jump frames are all held near ~27m, not ~37-39m


def test_jump_guard_self_heals_after_cap():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=20.0, vRel=-1.0, vLead=19.0)))
  for _ in range(JUMP_GUARD_MAX_HOLD):
    out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
    assert out.leadOne.dRel < 40.0                   # held while under the cap
  # cap just reached on a lead that was closing -- one bounded grace cycle before accepting a farther raw value
  out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
  assert out.leadOne.dRel < 40.0                     # grace cycle: still held, not yet accepted
  out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
  assert out.leadOne.dRel == pytest.approx(40.0)     # grace spent -> accepts the real (departing) value


def test_jump_guard_self_heals_immediately_when_not_closing():
  # The grace cycle only protects a lead that was closing when the cap was hit -- a lead that was already
  # steady/opening (vRel >= 0) self-heals on the very first cap-exceeding frame, same as before this fix.
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=20.0, vRel=0.5, vLead=19.0)))
  for _ in range(JUMP_GUARD_MAX_HOLD):
    out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=0.5, vLead=19.0)))
    assert out.leadOne.dRel < 40.0
  out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=0.5, vLead=19.0)))
  assert out.leadOne.dRel == pytest.approx(40.0)     # no grace needed -> heals immediately, unchanged behavior


def test_jump_guard_grace_is_used_at_most_once_per_hold_episode():
  # The grace cycle must be bounded -- a lead that keeps reading farther after the grace is spent must not
  # get a second grace before genuinely accepting the new value (else a departed lead could be held forever).
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=20.0, vRel=-1.0, vLead=19.0)))
  for _ in range(JUMP_GUARD_MAX_HOLD):
    c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
  c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))    # grace cycle, spent
  out = c.smooth_radarstate(rs(lead(dRel=70.0, vRel=-1.0, vLead=19.0)))
  assert out.leadOne.dRel == pytest.approx(70.0)     # grace already spent this episode -> accepts immediately


def test_jump_guard_replays_real_route_dropout_catchup():
  # route 550a71ee4c7a7fbe/000004c6--ed1b6d7f95, t~1337.9-1338.5: a spurious closer misread (31.08 -> 24.94)
  # passes through immediately (closer always does), poisoning the guard's anchor. The lead's real, continuing
  # trajectory (~31m, closing) then reads as a farther jump against that bad anchor and gets held for the full
  # cap. Without the grace cycle, the guard self-healed straight onto a farther transitional misread (56.52)
  # right as a real dropout began, and _LeadHold then flicker-held THAT value through the whole dropout --
  # reporting a lead ~2x farther and opening instead of closing, easing the MPC off right before a real
  # catch-up brake. The grace cycle must keep the held value close to the real trajectory across this handoff.
  c = ctrl(v_ego=14.4)
  raw = [
    (30.62, -0.45, 1), (38.36, -3.33, -1), (38.20, -3.42, -1), (38.08, -3.45, -1), (37.88, -3.53, -1),
    (37.72, -3.58, -1), (47.53, 0.35, -1), (24.94, -1.90, -1), (31.44, -3.72, -1), (31.20, -3.97, -1),
    (31.08, -3.95, 2), (74.32, 3.65, 3), (74.52, 3.70, 3), (74.92, 3.85, 3), (75.12, 3.88, 3),
    (75.28, 3.90, 3), (75.64, 3.95, 3), (75.64, 3.95, 3), (56.52, -2.09, -1),
  ]
  out = None
  for dRel, vRel, tid in raw:
    out = c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=10.5, radarTrackId=tid)))
  assert out.leadOne.dRel < 30.0                     # grace cycle: still held near the real trajectory
  dropout_held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert dropout_held.status is True
  assert dropout_held.dRel < 30.0                    # flicker-hold seeds from the grace-held value, not 56.52


def test_jump_guard_resets_on_dropout():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=20.0, vRel=-1.0, vLead=19.0)))
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
  assert out.leadOne.dRel == pytest.approx(40.0)     # a real dropout in between is not a same-cycle jump


def test_jump_guard_off_when_disabled():
  c = ctrl(enabled=False)
  c.smooth_radarstate(rs(lead(dRel=27.92, vRel=-5.60, vLead=24.47)))
  r = rs(lead(dRel=38.88, vRel=-3.19, vLead=26.91))
  assert c.smooth_radarstate(r) is r                 # disabled -> raw passthrough, no guard


def test_jump_guard_replays_real_route_sub_threshold_bounce():
  # route 550a71ee4c7a7fbe/000004b4--2bd66184db, t~976.08-976.48: dRel bounced 17.70 -> 12.32 -> ... -> 17.15
  # -> 12.04m across ~0.4s while vRel stayed -0.8 to -2.4 m/s -- physically impossible for one real object
  # at that closing speed (5m in ~0.1s would need ~50 m/s, not ~1-2). This is the case that motivated
  # lowering SWITCH_DREL from 8.0 to 4.0: the farther excursion (12.24 -> 17.15, a 4.91m jump) sailed through
  # unguarded at the old threshold, producing a false-relief-then-correction whipsaw. A closer jump (e.g.
  # 17.70 -> 12.32) always passes immediately regardless of threshold -- that invariant is untouched here.
  c = ctrl()
  raw = [
    (19.12, -2.32, 9.22, -0.67, -1), (17.95, -2.06, 9.39, -0.58, -1), (18.06, -1.90, 9.49, -0.60, -1),
    (17.70, -1.84, 9.44, -0.52, -1), (12.32, -1.20, 10.01, -0.02, 2449), (12.12, -1.40, 9.75, -1.60, 2427),
    (12.56, -1.20, 9.87, -1.45, 2427), (12.24, -1.05, 9.92, -1.29, 2427), (17.15, -2.39, 8.53, -0.85, -1),
    (12.04, -0.82, 10.02, -0.97, 2427), (12.04, -0.82, 9.94, -0.85, 2427), (11.80, -0.85, 9.81, -0.78, 2427),
  ]
  out = None
  seen = []
  for dRel, vRel, vLead, aLeadK, tid in raw:
    out = c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=vLead, aLeadK=aLeadK, radarTrackId=tid)))
    seen.append(out.leadOne.dRel)
  assert seen[4] == pytest.approx(12.32)          # the initial closer jump (17.70->12.32) passes immediately
  assert seen[8] < 14.0                            # the 12.24->17.15 farther excursion is held, not passed
  assert out.leadOne.dRel == pytest.approx(11.80)  # recovers exactly once raw resumes reporting close values


def test_jump_guard_hold_caps_model_prob_for_fcw():
  # route 550a71ee4c7a7fbe/000004bc--d9e0efd5ac, t~1563.5: a real, high-confidence (modelProb 0.999) lead
  # departs and gets held near-stationary by the guard mid-launch. The stock crash_cnt FCW gate fires on
  # radarState.leadOne.modelProb > 0.9 -- a held (stale, no longer confirmed-fresh) reading must not carry
  # enough confidence on its own to satisfy that gate, matching _LeadHold's existing flicker-hold cap.
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=15.72, vRel=1.80, vLead=6.79, modelProb=0.999)))
  held = c.smooth_radarstate(rs(lead(dRel=15.72 + SWITCH_DREL + 1.0, vRel=6.5, vLead=6.8, modelProb=0.999))).leadOne
  assert held.modelProb <= FCW_PROB_CAP


def test_jump_guard_boundary_not_triggered():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-2.0, vLead=18.0)))
  out = c.smooth_radarstate(rs(lead(dRel=30.0 + SWITCH_DREL - 0.1, vRel=-2.0, vLead=18.0)))
  assert out.leadOne.dRel == pytest.approx(30.0 + SWITCH_DREL - 0.1)   # under threshold -> passes through


def test_jump_guard_does_not_hold_a_stale_reference_after_an_extended_low_speed_gap():
  # Mirrors _LeadHold's identical bug (see test_hold_does_not_resurrect_a_stale_lead_after_an_extended_low_speed_gap):
  # _jump_guard.step() is also only called above LOW_SPEED_PASSTHROUGH_V, so its _last reference used to
  # freeze for the entire duration of any low-speed period with no elapsed-time awareness. Route
  # 550a71ee4c7a7fbe/000004dc--c8c0867520, t~407.1: a lead tracked at dRel=11.72 while decelerating into a
  # stop froze there through a ~160s standstill. On relaunch the real lead (now dRel=23.16, opening) was
  # diffed against that 160s-stale reference as if it were a same-cycle transient, held as a fabricated
  # closing lead, and fed the MPC a phantom near-collision course that produced a real, unwarranted hard
  # brake on a real drive.
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  c.smooth_radarstate(rs(lead(dRel=11.72, vRel=-2.33, vLead=2.72)))    # last reading before decelerating to a stop
  c._v_ego = CREEP_PASSTHROUGH_V - 0.5                                 # full stop: jump-guard.step() never called
  for _ in range(HOLD_MAX_FRAMES * 10):                                # far longer than any hold cap -- a real gap
    c.smooth_radarstate(rs(lead(dRel=3.68, vRel=0.0, vLead=0.0)))     # the lead is stopped just ahead
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0                             # relaunch: real lead now far + opening
  out = c.smooth_radarstate(rs(lead(dRel=23.16, vRel=4.72, vLead=9.77))).leadOne
  assert out.dRel == pytest.approx(23.16)             # fresh reading passes through, not held as a phantom jump
  assert out.vRel == pytest.approx(4.72)


def test_jump_guard_survives_a_single_incidental_gap_without_losing_protection():
  # Regression: a flat ">1 frame gap -> stale" threshold discarded the trusted reference (and so skipped the
  # SWITCH_DREL check entirely) after ANY single skipped call -- not just a real multi-second stop. That
  # happens on every cycle of ordinary v_ego dithering right at LOW_SPEED_PASSTHROUGH_V (a car crawling near
  # 5 m/s in stop-and-go traffic), permanently voiding the same-cycle fusion-transient guard for as long as the
  # dithering continues and letting a real glitch straight through. A single one-frame dip (a momentary v_ego
  # dip below the gate, then immediately back above it) must NOT be treated as stale -- the original farther-
  # jump rejection must still fire right after it, same as it would with no gap at all.
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-2.0, vLead=18.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V - 1.0             # one incidental dip below the gate
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0             # immediately back above it
  glitch = rs(lead(dRel=30.0 + SWITCH_DREL + 1.0, vRel=6.0, vLead=20.0))   # same-cycle fusion transient
  out = c.smooth_radarstate(glitch).leadOne
  assert out.dRel < 32.0                               # still held near ~30, glitch did not leak through


# --- flicker-hold -----------------------------------------------------------------------------------------

def test_holds_after_sustained_dropout():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-4.0, vLead=16.0)))
  held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert held.status is True
  assert held.dRel < 30.0                            # dead-reckoned closer
  assert held.dRel == pytest.approx(30.0 - 4.0 * 0.05, abs=1e-6)


def test_no_hold_without_sustained_lead():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=30.0)))           # single frame < SUSTAIN_FRAMES
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is False                 # no hold armed


def test_hold_does_not_resurrect_a_stale_lead_after_an_extended_low_speed_gap():
  # Below LOW_SPEED_PASSTHROUGH_V the hold is never stepped at all (see smooth_radarstate), so an elapsed-
  # frames check must be based on real cycles, not "cycles since step() was last called" -- otherwise resuming
  # above the gate looks like no time passed no matter how long the low-speed period actually was, and a hold
  # armed on a real lead long before the gap can resurrect as if it were still fresh.
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-3.0, vLead=5.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V - 1.0             # below the gate: step() stops being called on the hold
  for _ in range(HOLD_MAX_FRAMES * 3):
    c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0             # back above the gate, lead still gone
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is False                  # must not resurrect the old hold


def test_hold_survives_a_brief_low_speed_dip_within_the_cap():
  # A short dip below the gate (well under HOLD_MAX_FRAMES real cycles) is the case flicker-hold exists for --
  # it must still bridge, same as a same-speed dropout of the same real duration would. dRel must stay close
  # to the real last-known value (a genuine dead-reckoned extrapolation) -- NOT collapse toward MIN_HELD_DREL,
  # which is what a broken reseed would produce (see test_hold_reseeds_correctly_after_any_low_speed_gap).
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-3.0, vLead=5.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V - 1.0
  for _ in range(3):
    c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is True
  assert out.leadOne.dRel < 30.0
  assert out.leadOne.dRel > 29.0                     # dead-reckoned from 30.0, not collapsed to MIN_HELD_DREL


def test_hold_reseeds_correctly_after_any_low_speed_gap():
  # Regression: _held_dRel used to be reseeded to the real last-known value only when since_real == 1. Once
  # since_real became elapsed-REAL-frames (not a self-incrementing call counter), any skipped low-speed frame
  # made since_real > 1 on the very first dropout call actually made, silently skipping the reseed -- leaving
  # _held_dRel at its stale/init value (0.0), which the very next line's floor clamps to MIN_HELD_DREL: a
  # fabricated near-bumper phantom lead fed straight to the MPC, not a dead-reckoned extrapolation of the real
  # one. A single one-frame low-speed dip (the shortest possible gap, ~0.05s) is enough to trigger it.
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=50.0, vRel=-3.0, vLead=15.0)))
  c._v_ego = CREEP_PASSTHROUGH_V - 0.5
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))   # exactly ONE low-speed frame
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert out.status is True
  assert out.dRel > 45.0                             # must be near the real ~50m, not a fabricated 0.5m


def test_releases_after_hold_cap():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-2.0)))
  drop = rs(lead(status=False, dRel=0.0, modelProb=0.0))
  for _ in range(HOLD_MAX_FRAMES):
    assert c.smooth_radarstate(drop).leadOne.status is True
  assert c.smooth_radarstate(drop).leadOne.status is False   # released after the cap


def test_obstacle_monotone_during_hold():
  c = ctrl()
  for _ in range(3):
    real = lead(dRel=30.0, vRel=-3.0, vLead=15.0)
    c.smooth_radarstate(rs(real))
  base = obstacle(real)
  drop = rs(lead(status=False, dRel=0.0, modelProb=0.0))
  prev = base
  for _ in range(HOLD_MAX_FRAMES):
    held = c.smooth_radarstate(drop).leadOne
    assert obstacle(held) <= prev + 1e-6             # never reports a farther obstacle -> brake >= stock
    prev = obstacle(held)


def test_fcw_prob_capped_and_aleadk_not_positive():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, aLeadK=1.5, modelProb=0.99)))
  held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert held.modelProb <= FCW_PROB_CAP
  assert held.aLeadK <= 0.0


def test_flicker_does_not_reset_wall_clock():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-2.0)))
  # alternating drop/reacquire must not refill the hold budget: after > HOLD_MAX_FRAMES wall time it releases
  for i in range(HOLD_MAX_FRAMES + 4):
    frame = rs(lead(status=False, dRel=0.0, modelProb=0.0)) if i % 2 else rs(lead(dRel=0.5))  # dRel<=DROPOUT: not real
    c.smooth_radarstate(frame)
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is False
  assert DROPOUT_DREL == 1.0


# --- churn smoother ---------------------------------------------------------------------------------------

def test_churn_smoother_removes_jitter():
  c = ctrl()
  out = None
  for f in churn_frames(30):
    out = c.smooth_radarstate(rs(f))
  assert c.lead_unstable()                           # churn detected
  assert 40.0 < out.leadOne.dRel < 42.0              # EMA settled between the two jittering tracks
  assert out.leadOne.dRel not in (40.0, 42.0)        # not the raw alternating value


def test_churn_smoother_off_when_disabled():
  c = ctrl(enabled=False)
  out = None
  for f in churn_frames(30):
    r = rs(f)
    out = c.smooth_radarstate(r)
    assert out is r                                  # disabled -> raw passthrough, no smoothing


def test_smoother_inactive_without_churn():
  c = ctrl()
  one = lead(dRel=40.0, radarTrackId=7)
  for _ in range(10):
    out = c.smooth_radarstate(rs(lead(dRel=40.0, radarTrackId=7)))
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one                          # steady id -> no churn -> exact passthrough


def test_churn_smoother_closer_accepted_immediately():
  # A steadily-closing lead that also briefly churns must never be held farther than the current raw value --
  # otherwise the EMA lags a real closing lead for the whole LEAD_SMOOTH_HOLD window, then snaps (a false
  # relief followed by a hard catch-up brake -- route 550a71ee4c7a7fbe/00000499, t~1387, real regression).
  c = ctrl()
  d = 82.0
  for i in range(40):
    tid = 1 if i % 3 else 2   # enough id-churn to keep the smoother engaged
    out = c.smooth_radarstate(rs(lead(dRel=d, vRel=-6.0, vLead=24.0, radarTrackId=tid)))
    assert out.leadOne.dRel <= d + 1e-6                # never farther than the latest raw reading
    d -= 0.4                                            # steadily closing


def test_churn_smoother_replays_real_route_late_acquisition():
  # route 550a71ee4c7a7fbe/00000499--7f57e1d000, t~1386.9-1388.4: radard toggles between two real candidate
  # tracks (id 4611 ~110m, id 4609 ~82m closing) while acquiring, then a couple of vision-fallback frames
  # (id -1) report ~104-109m mid-acquisition. The real dRel (track 4609) closes smoothly 82.0 -> 73.2m the
  # whole time. Old symmetric EMA held the reported dRel near ~82m (farther than truth) for ~1s after the
  # brief churn window, then snapped -- this is the false-relief-then-correction pattern being fixed here.
  c = ctrl()
  raw = [
    (110.84, -1.75, 4611), (82.04, -3.78, 4609), (110.60, -1.85, 4611), (81.68, -3.80, 4609),
    (82.28, -4.13, 4609), (110.40, -2.05, 4611), (110.16, -1.93, 4611), (110.08, -2.00, 4611),
    (110.00, -2.00, 4611), (80.12, -4.83, 4609), (79.88, -4.95, 4609), (79.64, -5.08, 4609),
    (79.32, -5.20, 4609), (79.48, -5.38, 4609), (79.08, -5.55, 4609), (78.64, -5.70, 4609),
    (78.20, -5.88, 4609), (77.84, -6.00, 4609), (77.60, -6.18, 4609), (77.48, -6.30, 4609),
    (76.96, -6.50, 4609), (76.48, -6.65, 4609), (103.52, -1.75, -1), (76.08, -6.90, 4609),
    (75.52, -7.05, 4609), (108.97, -2.02, -1), (104.23, -2.15, -1), (103.64, -2.13, -1),
    (74.16, -7.43, 4609), (73.72, -7.60, 4609), (73.24, -7.70, 4609),
  ]
  out = None
  for dRel, vRel, tid in raw:
    out = c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=24.0 + vRel, radarTrackId=tid)))
  assert out.leadOne.dRel == pytest.approx(73.24, abs=0.5)   # tracks the true closing value, no lag
  # at no point does the reported dRel sit meaningfully farther than the most recent real (id>0) reading
  c = ctrl()
  worst_overshoot = 0.0
  last_real = None
  for dRel, vRel, tid in raw:
    out = c.smooth_radarstate(rs(lead(dRel=dRel, vRel=vRel, vLead=24.0 + vRel, radarTrackId=tid)))
    if tid > 0:
      last_real = dRel
    if last_real is not None:
      worst_overshoot = max(worst_overshoot, out.leadOne.dRel - last_real)
  assert worst_overshoot < 1.0                              # old code overshot by ~6-9m for up to ~1s


# --- instability telemetry --------------------------------------------------------------------------------

def test_stability_quiet_on_clean_lead():
  c = ctrl()
  for _ in range(10):
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0, radarTrackId=5)))
  assert not c.lead_unstable()


def test_stability_flags_bimodal_lead():
  c = ctrl()
  for i in range(10):
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0 if i % 2 else 10.0, radarTrackId=5)))
  assert c.lead_unstable()


def test_stability_flags_trackid_churn():
  c = ctrl()
  for f in churn_frames(20):
    c.smooth_radarstate(rs(f))
  assert c.lead_unstable()


def test_stability_resets_on_dropout():
  c = ctrl()
  for i in range(10):
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0 if i % 2 else 10.0)))
  assert c.lead_unstable()
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert not c.lead_unstable()


def test_stability_runs_even_when_disabled():
  c = ctrl(enabled=False)
  for i in range(10):
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0 if i % 2 else 10.0)))
  assert c.lead_unstable()                           # telemetry not gated by the RadarDistance param


# --- same-track noise smoother (bimodal vLead / repeated dRel jump on a CONSTANT radarTrackId) -------------

def test_smoother_dejitters_bimodal_vlead_on_same_track():
  # Same physical object (radarTrackId constant) but a bouncing velocity read (Doppler/fusion noise) -- the
  # id evidence pins this to ONE real lead, so it's safe to EMA (unlike a bimodal read with a changing id).
  c = ctrl()
  out = None
  for i in range(30):
    out = c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0 if i % 2 else 10.0, vRel=-1.0, radarTrackId=9)))
  assert c.lead_unstable()
  assert 10.0 < out.leadOne.vLead < 18.0             # EMA settled between the two bouncing readings
  assert out.leadOne.vLead not in (10.0, 18.0)


def test_smoother_inactive_on_bimodal_vlead_with_changing_track():
  # Same bimodal vLead signature, but radarTrackId ALSO changes -- ambiguous (could be two really-different
  # real objects at different speeds), so this must NOT be smoothed, unlike the same-track case above.
  c = ctrl()
  one = lead(dRel=40.0, vLead=18.0, radarTrackId=1)
  for i in range(10):
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=18.0 if i % 2 else 10.0, radarTrackId=1 if i % 2 else 2)))
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one                          # exact passthrough -- not averaged across tracks


def test_smoother_same_track_noise_ignores_drel_jump():
  # dRel track-jumps are excluded from same_track_noise on purpose: while status stays True, a repeated
  # farther jump this large is already absorbed by _JumpGuard upstream, so the smoother never even sees the
  # raw alternation here -- confirms the two mechanisms don't double up on the same signal.
  c = ctrl()
  out = None
  for i in range(30):
    out = c.smooth_radarstate(rs(lead(dRel=40.0 if i % 2 == 0 else 55.0, vLead=18.0, vRel=-1.0, radarTrackId=4)))
  assert out.leadOne.dRel < 45.0                     # held near the trusted value by the jump-guard, not 55


def test_smoother_does_not_lag_a_stale_ema_after_an_extended_low_speed_gap():
  # Mirrors the identical bug already fixed in _JumpGuard/_LeadHold: _smoother.update() is only called above
  # CREEP_PASSTHROUGH_V (see smooth_radarstate), so its EMA state (_d/_vl/_vr) and _hold freeze for the entire
  # duration of any full standstill. Resuming and EMA-ing a real, opening lead against that frozen state as if
  # no time had passed lags dRel toward the stale, closer pre-stop value -- confirmed on the same real route as
  # the _JumpGuard bug (550a71ee4c7a7fbe/000004dc--c8c0867520): pre-fix this reported 12.86m instead of the
  # real 23.16m on relaunch.
  c = ctrl(v_ego=LOW_SPEED_PASSTHROUGH_V + 1.0)
  for i in range(12):
    c.smooth_radarstate(rs(lead(dRel=11.72, vLead=2.72, vRel=-2.33, radarTrackId=1 if i % 2 else 2)))  # arms churn
  c._v_ego = CREEP_PASSTHROUGH_V - 0.5                   # full stop: smoother.update() never called
  for _ in range(60):
    c.smooth_radarstate(rs(lead(dRel=3.68, vRel=0.0, vLead=0.0)))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 1.0               # relaunch: real lead now far + opening
  out = c.smooth_radarstate(rs(lead(dRel=23.16, vRel=4.72, vLead=9.77, radarTrackId=1))).leadOne
  assert out.dRel == pytest.approx(23.16)                # fresh reading passes through, not EMA-lagged stale


def test_single_incidental_gap_during_churn_does_not_leak_a_glitch_into_the_ema():
  # Regression, deeper than the single-mechanism cases above: a flat ">1 frame gap -> stale" threshold made
  # _jump_guard treat ANY single skipped call as fully stale and skip the SWITCH_DREL check -- so a same-cycle
  # fusion-transient glitch right after one incidental low-speed dip passed straight through _jump_guard
  # unguarded, then got folded into the churn smoother's EMA (which was still live from before the dip),
  # lagging vLead/dRel toward the glitch's inflated values for ~1s: a farther-and-faster-than-real lead, i.e.
  # a real violation of this file's own invariant ("NEVER report a farther-or-faster lead than reality").
  c = ctrl(v_ego=10.0)
  for i in range(6):                                     # prime churn (real radar trackId-flip signature)
    tid = 1 if i % 2 == 0 else 2
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-1.0, vLead=15.0, radarTrackId=tid)))
  c._v_ego = 3.0                                         # one incidental dip into the creep band
  c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-1.0, vLead=15.0, radarTrackId=1)))
  c._v_ego = 10.0                                        # immediately back above the full-pipeline gate
  glitch = lead(dRel=30.0 + SWITCH_DREL + 1.0, vRel=6.0, vLead=20.0, radarTrackId=1)
  out = c.smooth_radarstate(rs(glitch)).leadOne
  assert out.dRel < 31.0                                 # held near real ~30, not the glitch's 35.0
  assert out.vLead < 16.0                                # held near real ~15, not the glitch's 20.0
