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
  out = c.smooth_radarstate(rs(lead(dRel=40.0, vRel=-1.0, vLead=19.0)))
  assert out.leadOne.dRel == pytest.approx(40.0)     # cap exceeded -> accepts the real (departing) value


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


def test_jump_guard_boundary_not_triggered():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-2.0, vLead=18.0)))
  out = c.smooth_radarstate(rs(lead(dRel=30.0 + SWITCH_DREL - 0.1, vRel=-2.0, vLead=18.0)))
  assert out.leadOne.dRel == pytest.approx(30.0 + SWITCH_DREL - 0.1)   # under threshold -> passes through


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
