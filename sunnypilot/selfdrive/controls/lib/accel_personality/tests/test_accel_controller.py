"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  ECO, NORMAL, SPORT, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, RISE_RATE, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, HARD_BRAKE_TARGET_ACCEL, OVERBITE_CAP, \
  STOP_PASSTHROUGH_V, ONSET_SPREAD_MAX, AccelerationPersonality

T_IDXS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
_EPS = 1e-6


class FakeParams:
  def __init__(self, store=None):
    self.store = dict(store or {})

  def get_bool(self, key):
    return bool(self.store.get(key, False))

  def get(self, key, return_default=False):
    return int(self.store.get(key, 1))

  def put(self, key, val, block=False):
    self.store[key] = val


def make_sm(v_ego=20.0, lead_status=False, lead_d=0.0, lead_vlead=0.0):
  lead = SimpleNamespace(status=lead_status, dRel=lead_d, vLead=lead_vlead)
  return {'carState': SimpleNamespace(vEgo=v_ego), 'radarState': SimpleNamespace(leadOne=lead)}


def make_controller(enabled=True, personality=NORMAL, crash_cnt=0, comfort_stop=False):
  store = {"AccelPersonalityEnabled": enabled, "AccelPersonality": int(personality)}
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(crash_cnt=crash_cnt), params=FakeParams(store))
  ctrl._comfort_stop_enabled = comfort_stop   # comfort_stop is gated off in production; opt in per-test
  ctrl.update(make_sm())
  return ctrl


def flat_traj(value):
  return [float(value)] * len(T_IDXS)


# --- Profiles / off==stock ---------------------------------------------------

def test_enum_source_parity():
  assert (ECO, NORMAL, SPORT) == (AccelerationPersonality.eco, AccelerationPersonality.normal, AccelerationPersonality.sport)
  assert (PERSONALITY_MIN, PERSONALITY_MAX) == (0, 2)


def test_disabled_forces_normal_and_stock_ceiling():
  ctrl = make_controller(enabled=False, personality=SPORT)
  assert ctrl.personality() == NORMAL
  assert not ctrl.enabled()
  for v in (0.0, 10.0, 25.0, 40.0):
    assert ctrl.get_max_accel(v) == pytest.approx(np.interp(v, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert ctrl.get_rise_rate() == STOCK_RISE_RATE


def test_disabled_passes_brake_through():
  ctrl = make_controller(enabled=False)
  for raw in (-3.0, -1.5, -0.5, 0.0, 1.0):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)


def test_normal_is_distinct_from_stock():
  # off==stock is enforced via the disabled path, NOT by NORMAL==stock, so enabled NORMAL is free to differ.
  ctrl = make_controller(personality=NORMAL)
  assert ctrl.get_max_accel(0.0) != pytest.approx(np.interp(0.0, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert ctrl.get_rise_rate() != STOCK_RISE_RATE


def test_ceiling_ordering_eco_lt_normal_lt_sport():
  eco, normal, sport = (make_controller(personality=p) for p in (ECO, NORMAL, SPORT))
  for v in (0.0, 14.0, 25.0, 40.0):
    assert eco.get_max_accel(v) < normal.get_max_accel(v) < sport.get_max_accel(v)
  assert eco.get_rise_rate() < normal.get_rise_rate() < sport.get_rise_rate()


def test_rise_rate_ordering():
  assert RISE_RATE[ECO] < RISE_RATE[NORMAL] < RISE_RATE[SPORT]


# --- SAFETY: never weaker than the plan, hard brakes never delayed --------------

@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_than_plan_sustained(personality):
  # Safety: an EMERGENCY brake is never weaker than the plan (strict). A non-emergency brake may lag the plan
  # by at most ONSET_SPREAD_MAX (the bounded onset-spread) and no more.
  ctrl = make_controller(personality=personality)
  for raw in [0.0, -0.2, -0.5, -0.9, -1.2, -1.5, -2.0] + [-2.0] * 20:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    if raw <= HARD_BRAKE_TARGET_ACCEL:
      assert out <= raw + _EPS                                 # emergency: strict never-weaker
    elif raw < 0.0:
      assert out <= raw + ONSET_SPREAD_MAX + _EPS              # non-emergency: bounded onset-spread only


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_random_walk(personality):
  rng = np.random.default_rng(0)
  ctrl = make_controller(personality=personality)
  for _ in range(500):
    raw = float(rng.uniform(-2.5, 1.5))
    traj = flat_traj(raw - float(rng.uniform(0.0, 0.6)))
    out = ctrl.smooth_target_accel(raw, traj, T_IDXS, should_stop=False)
    if raw < 0.0:
      assert out <= raw + ONSET_SPREAD_MAX + _EPS              # never more than the bounded onset-spread weaker


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_hard_brake_passes_through_immediately(personality):
  # Regression for route 00000466 near-crash: a sudden hard brake (plan steps deep) must reach FULL depth
  # on the FIRST frame -- never rate-limited / delayed, or the car under-brakes into a closing lead.
  ctrl = make_controller(personality=personality)
  out = ctrl.smooth_target_accel(-3.5, flat_traj(-3.5), T_IDXS, should_stop=False)
  assert out == pytest.approx(-3.5, abs=_EPS)
  assert ctrl.bypassed()


def test_sudden_lead_no_brake_delay():
  # The exact 466 shape: cruising (plan +1.7, no brake) then a fast lead appears and the plan steps to max
  # brake. The commanded brake must hit full depth immediately, not ramp in over time.
  ctrl = make_controller(personality=ECO)
  for _ in range(5):
    ctrl.smooth_target_accel(1.7, flat_traj(1.7), T_IDXS, should_stop=False)   # cruising, no lead
  out = ctrl.smooth_target_accel(-3.5, flat_traj(-3.5), T_IDXS, should_stop=False)  # lead appears
  assert out == pytest.approx(-3.5, abs=_EPS)                                  # full brake, zero delay


def test_should_stop_passes_through():
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=True)
  assert out == pytest.approx(-1.0, abs=_EPS)
  assert ctrl.bypassed()


def test_fcw_crash_passes_through():
  ctrl = make_controller(personality=ECO, crash_cnt=3)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-1.0, abs=_EPS)
  assert ctrl.bypassed()


def test_blended_never_weaker():
  # Blended/e2e (stock_brake): never weaker than the plan (may anticipate via the never-weaker front-load).
  ctrl = make_controller(personality=ECO)
  for raw in [0.0, -0.3, -0.6, -0.9, -1.0, -1.0, -1.0]:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False, stock_brake=True)
    assert out <= raw + _EPS


# --- Anticipatory front-load (never weaker, capped) ------------------------------

def test_front_load_brakes_before_plan():
  # A deeper brake is predicted ahead (brake_need=1.0) while the live plan is still flat -> front-load
  # brakes early (output goes negative), but the smooth branch keeps it never weaker than the plan.
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(0.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out < 0.0
  assert ctrl.smooth_active()
  assert ctrl.brake_need() == pytest.approx(1.0)


def test_front_load_anticipates_below_live_plan():
  # When the live plan is gently braking and a deeper brake is predicted, the front-load deepens below the
  # live plan (anticipatory early brake), settling within OVERBITE_CAP of it.
  ctrl = make_controller(personality=ECO)
  out = 0.0
  for _ in range(20):
    out = ctrl.smooth_target_accel(-0.2, flat_traj(-1.5), T_IDXS, should_stop=False)
  assert out < -0.2 - _EPS                                   # deeper than the live -0.2 plan
  assert out >= -0.2 - OVERBITE_CAP - _EPS                   # but never more than the cap below it


def test_overbite_cap_limits_frontload_vs_live_plan():
  # Cut-in/merge: plan still wants throttle (+0.5) while a deep brake is predicted -> front-load may not
  # settle more than OVERBITE_CAP below the live plan (no abrupt early over-bite).
  ctrl = make_controller(personality=ECO)
  traj = [0.5, 0.3, 0.0, -0.5, -1.5, -2.0] + [-2.0] * (len(T_IDXS) - 6)
  out = 0.0
  for _ in range(10):
    out = ctrl.smooth_target_accel(0.5, traj, T_IDXS, should_stop=False)
  assert ctrl.smooth_active()
  assert out == pytest.approx(0.5 - OVERBITE_CAP, abs=1e-3)


# --- Stop / low-speed neutrality -------------------------------------------------

def test_low_speed_brake_is_stock_passthrough():
  # Stop/creep regime (vEgo < STOP_PASSTHROUGH_V): braking is stock so the stop distance matches OFF.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  for raw in (-0.3, -1.0):
    out = ctrl.smooth_target_accel(raw, flat_traj(-1.5), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)
    assert not ctrl.smooth_active()


def test_low_speed_launch_still_shapes():
  # The low-speed brake passthrough must NOT neutralize positive-accel (launch) shaping.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)
  out = ctrl.smooth_target_accel(1.5, flat_traj(1.5), T_IDXS, should_stop=False)
  assert out < 1.5                                           # rise-rate limited (shaped)


def test_stop_imminent_passthrough_but_moving_follow_shapes():
  # Stop coming (plan speed -> ~0): stock passthrough (no coast-in). Slowing to a moving follow: front-load
  # stays active so the early-brake goal is preserved.
  ctrl = make_controller(personality=ECO)
  stopping = [3.0, 2.0, 1.0, 0.4, 0.0] + [0.0] * (len(T_IDXS) - 5)
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=stopping)
  assert not ctrl.smooth_active()
  assert out == pytest.approx(-0.1, abs=_EPS)
  moving = [8.0] * len(T_IDXS)
  ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=moving)
  assert ctrl.smooth_active()


def test_comfort_stop_holds_through_plan_ease():
  # Plan brakes to a peak then eases off near the stop (the stock creep). The hold keeps the deeper decel so
  # the brake does not ease in (no roll) -- but NEVER firmer than the plan's own peak (no added hard bite).
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  out = 0.0
  for plan in [-0.4, -0.8, -1.1, -1.1, -0.6, -0.3, -0.1]:   # decel to a -1.1 peak, then ease (creep)
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=6.0, lead_vlead=0.0))
    out = ctrl.smooth_target_accel(plan, flat_traj(plan), T_IDXS, should_stop=False)
  assert out < -0.3 - _EPS                              # held deeper than the easing plan (-0.1) -> no creep-in
  assert out >= -1.1 - _EPS                             # but never firmer than the plan's own peak (no -1.6 bite)


def test_comfort_stop_never_firmer_than_plan():
  # The hold can only stop the brake from WEAKENING; it never commands a decel firmer than the plan itself.
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  for plan in [-0.2, -0.5, -0.9, -0.9, -0.9]:           # steady (no ease) -> hold matches plan, adds nothing
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=6.0, lead_vlead=0.0))
    out = ctrl.smooth_target_accel(plan, flat_traj(plan), T_IDXS, should_stop=False)
    assert out == pytest.approx(plan, abs=_EPS)         # never firmer than the (non-easing) plan -> no bite/grab


def test_comfort_stop_monotone_no_early_release():
  # While still moving, the comfort floor never WEAKENS frame-to-frame (the old enforcer self-released -> roll).
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  floors = []
  for v in [3.0, 2.6, 2.2, 1.8, 1.4, 1.0, 0.6]:         # decelerating toward the lead
    ctrl.update(make_sm(v_ego=v, lead_status=True, lead_d=max(0.5, 7.0 - (3.0 - v) * 2), lead_vlead=0.0))
    ctrl.smooth_target_accel(-0.5, flat_traj(-0.5), T_IDXS, should_stop=False)
    floors.append(ctrl._stop_floor)
  for a, b in zip(floors, floors[1:], strict=False):
    assert b <= a + _EPS                                # monotone non-weakening while approaching


def test_comfort_stop_off_when_disabled():
  ctrl = make_controller(enabled=False, personality=ECO)
  ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=4.0, lead_vlead=0.0))
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False)
  assert out == pytest.approx(-0.1, abs=_EPS)


def test_comfort_stop_gated_off_is_stock_passthrough():
  # Production default (COMFORT_STOP_ENABLED off, even with AccelController enabled): the final approach is stock
  # passthrough -- output follows the easing plan, no anti-creep hold, floor stays 0 (goal 6 met by stock).
  ctrl = make_controller(personality=ECO)                      # comfort_stop defaults False (production)
  out = 0.0
  for plan in [-0.4, -0.8, -1.1, -0.6, -0.1]:                  # decel to a peak then ease (stock creep)
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=6.0, lead_vlead=0.0))
    out = ctrl.smooth_target_accel(plan, flat_traj(plan), T_IDXS, should_stop=False)
  assert out == pytest.approx(-0.1, abs=_EPS)                  # follows the easing plan -> no hold
  assert ctrl._stop_floor == 0.0                               # never latched


def test_comfort_stop_no_op_moving_lead():
  # Moving lead (vLead high): no comfort stop (only behind a near-stopped lead).
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=6.0, lead_vlead=5.0))
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-0.1), T_IDXS, should_stop=False)
  assert out == pytest.approx(-0.1, abs=_EPS)


def test_comfort_stop_never_weaker():
  # The comfort floor only ever ADDS braking: output never weaker than the plan.
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  for raw in (-0.05, -0.3, -1.0, -2.5):
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=5.5, lead_vlead=0.0))
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert out <= raw + _EPS


def test_comfort_stop_weakens_when_gap_opens():
  # Creeping stop-and-go lead (vLead stays < COMFORT_STOP_LEAD_V) that pulls away: once the gap opens well past
  # the target the floor must WEAKEN, not hold a phantom brake into an opening gap.
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  for _ in range(15):                                          # approach close -> deep floor (final-approach hold)
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=5.5, lead_vlead=0.3))
    ctrl.smooth_target_accel(-0.5, flat_traj(-0.5), T_IDXS, should_stop=False)
  deep = ctrl._stop_floor
  assert deep < -0.3
  for _ in range(25):                                          # lead creeps away (still vLead<1): gap opens wide
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=12.0, lead_vlead=0.5))
    ctrl.smooth_target_accel(-0.05, flat_traj(-0.05), T_IDXS, should_stop=False)
  assert ctrl._stop_floor > deep + 0.3                         # floor weakened as the gap opened (no phantom brake)


def test_comfort_stop_releases_on_launch():
  # Stop-and-go GO: after holding a comfort floor at a stop, once the lead moves and the plan wants throttle the
  # floor must release (track the plan up) and not hold the output below the natural plan -> the car launches.
  ctrl = make_controller(personality=ECO, comfort_stop=True)
  for _ in range(20):                                          # hold the plan's -1.0 decel approaching a stopped lead
    ctrl.update(make_sm(v_ego=1.5, lead_status=True, lead_d=6.0, lead_vlead=0.0))
    ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert ctrl._stop_floor < -0.5                               # floor holds the plan's decel (engaged/deep)
  out = 0.0
  for _ in range(30):                                          # lead launches, plan wants throttle
    ctrl.update(make_sm(v_ego=2.0, lead_status=True, lead_d=8.0, lead_vlead=4.0))
    out = ctrl.smooth_target_accel(0.8, flat_traj(0.8), T_IDXS, should_stop=False)
  assert out > 0.0                                             # launches (floor did not hold it back)
  assert ctrl._stop_floor == 0.0                               # floor fully released


def test_onset_spread_bounded_and_skipped_for_emergency():
  # Non-emergency brake onset is spread (lagged) but never by more than ONSET_SPREAD_MAX; an emergency brake
  # is instant full depth (no spread).
  ctrl = make_controller(personality=ECO)
  for _ in range(3):
    ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=False)   # non-emergency step
  assert out > -1.0 + _EPS                              # lagged (spread), not an instant step
  assert out <= -1.0 + ONSET_SPREAD_MAX + _EPS          # but bounded
  ctrl2 = make_controller(personality=ECO)
  for _ in range(3):
    ctrl2.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)
  out2 = ctrl2.smooth_target_accel(-2.0, flat_traj(-2.0), T_IDXS, should_stop=False)  # emergency (<= -1.5)
  assert out2 == pytest.approx(-2.0, abs=_EPS)


def test_disabled_hard_brake_is_instant_stock():
  ctrl = make_controller(enabled=False, personality=ECO)
  out = ctrl.smooth_target_accel(-3.0, flat_traj(-3.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-3.0, abs=_EPS)


# --- Misc ------------------------------------------------------------------------

def test_out_of_range_personality_clamps():
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(crash_cnt=0),
                         params=FakeParams({"AccelPersonalityEnabled": True, "AccelPersonality": 99}))
  ctrl.update(make_sm())
  assert ctrl.personality() == PERSONALITY_MAX


def test_reset_passes_through():
  ctrl = make_controller(personality=ECO)
  out = ctrl.smooth_target_accel(0.0, flat_traj(-1.0), T_IDXS, should_stop=False, reset=True)
  assert out == pytest.approx(0.0, abs=_EPS)
  assert not ctrl.bypassed()
