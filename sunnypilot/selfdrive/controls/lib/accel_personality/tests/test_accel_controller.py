"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController as _AC  # noqa: F401
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  ECO, NORMAL, SPORT, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, RISE_RATE, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_ONSET_JERK, OVERBITE_CAP, \
  STOP_PASSTHROUGH_V, AccelerationPersonality, \
  BRAKE_DEEPENING_JERK, ONSET_JERK0, ONSET_GAP_SOFT, ONSET_HANDBACK_JERK

# The convex onset brakes shallower than the plan during the bite, but the instantaneous-gap catch-up
# bounds how far it can lag, and it converges to the plan. The integrated velocity deficit over an
# armed brake stays under this conservative cap (no permanent offset; added stopping distance bounded).
_ONSET_VDEBT_BOUND = {ECO: 0.55, SPORT: 0.50}   # raised for the firm-bypass onset jerk cap (still bounded -> no runaway)

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


def make_sm(v_ego=20.0):
  return {'carState': SimpleNamespace(vEgo=v_ego)}


def make_controller(enabled=True, personality=NORMAL, crash_cnt=0):
  store = {"AccelPersonalityEnabled": enabled, "AccelPersonality": int(personality)}
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(crash_cnt=crash_cnt), params=FakeParams(store))
  ctrl.update(make_sm())
  return ctrl


def flat_traj(value):
  return [float(value)] * len(T_IDXS)


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
  for raw in (-1.5, -0.5, 0.0, 1.0):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)


def test_normal_is_distinct_from_stock():
  # off==stock is enforced via the disabled path (see test_disabled_forces_normal_and_stock_ceiling), NOT by
  # NORMAL==stock. So the enabled NORMAL tier is free to differ from stock -- and now does.
  ctrl = make_controller(personality=NORMAL)
  assert ctrl.get_max_accel(0.0) != pytest.approx(np.interp(0.0, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert ctrl.get_rise_rate() != STOCK_RISE_RATE


def test_ceiling_ordering_eco_lt_normal_lt_sport():
  # All three tiers are distinct: ECO < NORMAL < SPORT at every speed (launch and cruise). Each launches
  # promptly (peak + rise rate above stock), stepped by tier.
  eco, normal, sport = (make_controller(personality=p) for p in (ECO, NORMAL, SPORT))
  for v in (0.0, 14.0, 25.0, 40.0):
    assert eco.get_max_accel(v) < normal.get_max_accel(v) < sport.get_max_accel(v)
  assert eco.get_rise_rate() < normal.get_rise_rate() < sport.get_rise_rate()


def test_rise_rate_ordering():
  # ECO rise == stock (prompt launch ramp) by design; SPORT firmer than both.
  assert RISE_RATE[ECO] <= RISE_RATE[NORMAL] < RISE_RATE[SPORT]


def test_early_soft_braking_brakes_before_plan():
  ctrl = make_controller(personality=NORMAL)
  out = ctrl.smooth_target_accel(0.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out < 0.0
  assert ctrl.smooth_active()
  assert ctrl.brake_need() == pytest.approx(1.0)


def test_low_speed_brake_is_stock_passthrough():
  # Stop/creep regime (vEgo < STOP_PASSTHROUGH_V): braking is handed through unshaped (stock) so the
  # controller cannot soften the crawl and let the car coast in closer than stock.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  for raw in (-0.3, -1.0):                                   # a braking plan that would normally front-load
    out = ctrl.smooth_target_accel(raw, flat_traj(-1.5), T_IDXS, should_stop=False)
    assert out == pytest.approx(raw, abs=_EPS)              # exact stock passthrough, no front-load
    assert not ctrl.smooth_active()


def test_low_speed_launch_still_shapes():
  # The low-speed brake passthrough must NOT neutralize positive-accel (launch) shaping.
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=STOP_PASSTHROUGH_V - 0.1))
  ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)   # seed
  out = ctrl.smooth_target_accel(1.5, flat_traj(1.5), T_IDXS, should_stop=False)
  assert out < 1.5                                           # rise-rate limited (shaped), not raw passthrough


def test_overbite_cap_limits_frontload_vs_live_plan():
  # Cut-in/merge: raw plan still wants throttle (+0.5) while a deep brake is predicted ahead (brake_need
  # high). The front-load must not settle more than OVERBITE_CAP below the LIVE plan -> no abrupt early grab.
  ctrl = make_controller(personality=ECO)
  traj = [0.5, 0.3, 0.0, -0.5, -1.5, -2.0] + [-2.0] * (len(T_IDXS) - 6)   # throttle now, hard brake later
  out = 0.0
  for _ in range(8):                                       # let the accel rise-rate slew settle
    out = ctrl.smooth_target_accel(0.5, traj, T_IDXS, should_stop=False)
  assert ctrl.smooth_active()
  assert out == pytest.approx(0.5 - OVERBITE_CAP, abs=1e-3)  # front-load clamped to exactly cap below plan


def test_overbite_cap_preserves_anticipation_when_plan_braking():
  # Once the live plan is itself braking, raw-OVERBITE_CAP sits below the table, so the cap does NOT bind
  # and the anticipatory front-load (route 456 fix) is preserved (output deeper than the shallow raw).
  ctrl = make_controller(personality=ECO)
  out = 0.0
  for _ in range(4):
    out = ctrl.smooth_target_accel(-0.2, flat_traj(-1.5), T_IDXS, should_stop=False)
  assert out < -0.2 - _EPS                                 # still front-loads below the live -0.2 plan


def test_stop_imminent_stands_down_but_moving_follow_shapes():
  # Stop coming (plan speed -> ~0): stand down to stock decel so the gentle bite can't coast into the
  # stop (creep). Slowing to a MOVING follow (plan stays > STOP_IMMINENT_VEGO): gentle onset stays active
  # at every speed -> the gentle-brake goal is not regressed.
  ctrl = make_controller(personality=ECO)
  stopping = [3.0, 2.0, 1.0, 0.4, 0.0] + [0.0] * (len(T_IDXS) - 5)
  out = ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=stopping)
  assert not ctrl.smooth_active()
  assert out == pytest.approx(-0.1, abs=_EPS)             # stock passthrough into the stop, no softening
  moving = [8.0] * len(T_IDXS)                            # slowing to a moving follow, not a stop
  ctrl.smooth_target_accel(-0.1, flat_traj(-1.0), T_IDXS, should_stop=False, speed_trajectory=moving)
  assert ctrl.smooth_active()                             # gentle onset preserved (not stop-imminent)


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_than_plan_sustained_closing(personality):
  # NORMAL/off: strict never-weaker (route 000003da regression guard). ECO/SPORT: the convex onset may
  # lag the plan during the bite, but the INTEGRATED velocity deficit vs the plan stays bounded. This
  # sequence also crosses the -1.5 emergency bypass (bit-exact raw thereafter).
  ctrl = make_controller(personality=personality)
  vdebt = 0.0
  for raw in [0.0, -0.2, -0.5, -0.9, -1.2, -1.5] + [-1.5] * 40:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    if personality == NORMAL:
      assert out <= raw + _EPS
    else:
      vdebt = max(0.0, vdebt + (out - raw) * DT_MDL)
      assert vdebt <= _ONSET_VDEBT_BOUND[personality]


@pytest.mark.parametrize("personality", [ECO, NORMAL, SPORT])
def test_never_weaker_random_walk(personality):
  # NORMAL strict never-weaker; ECO/SPORT integrated velocity deficit stays bounded even under an
  # unrate-limited random plan (the deficit can never run away).
  rng = np.random.default_rng(0)
  ctrl = make_controller(personality=personality)
  vdebt = 0.0
  for _ in range(500):
    raw = float(rng.uniform(-1.9, 1.5))
    traj = flat_traj(raw - float(rng.uniform(0.0, 0.6)))
    out = ctrl.smooth_target_accel(raw, traj, T_IDXS, should_stop=False)
    if personality == NORMAL:
      if raw < 0.0:
        assert out <= raw + _EPS
    else:
      vdebt = max(0.0, vdebt + (out - raw) * DT_MDL)
      assert vdebt <= _ONSET_VDEBT_BOUND[personality]


def test_normal_brake_bit_exact_vs_legacy_slew():
  # NORMAL must be byte-identical to the legacy constant-jerk slew + min(slewed,raw) on a deepening ramp.
  ctrl = make_controller(personality=NORMAL)
  jmax = BRAKE_DEEPENING_JERK[NORMAL]
  last = 0.0
  for raw in [0.0, -0.1, -0.3, -0.45, -0.6, -0.6, -0.6, -0.5, -0.3, 0.0, 0.5]:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    if raw <= last:               # deepening: legacy step-limited then min(.,raw)
      legacy = min(max(raw, last - jmax * DT_MDL), raw) if raw < 0.0 else max(raw, last - jmax * DT_MDL)
      assert out == pytest.approx(legacy, abs=_EPS)
    last = out


@pytest.mark.parametrize("personality", [ECO, SPORT])
def test_convex_onset_gentle_bite(personality):
  # At a real brake onset the plan deepens gradually, so the gap stays within ONSET_GAP_SOFT and the
  # first deepening tick must use only the gentle initial jerk ONSET_JERK0 (the soft bite), NOT bite hard.
  ctrl = make_controller(personality=personality)
  ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)   # seed last=0
  raw = -0.5 * ONSET_GAP_SOFT[personality]   # within the gap budget -> gentle bite, no catch-up
  out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
  j_init = abs(out - 0.0) / DT_MDL           # realized first-tick jerk == ONSET_JERK0 (gentle bite)
  assert j_init == pytest.approx(ONSET_JERK0[personality], abs=1e-6)
  assert out > raw                           # softened: shallower than the plan, NOT passed through


@pytest.mark.parametrize("personality", [ECO, SPORT])
def test_convex_onset_velocity_deficit_bounded_and_converges(personality):
  # Integrated velocity deficit vs plan over a sustained moderate brake stays bounded, and the controller
  # CONVERGES to the plan (no permanent velocity offset -> added stopping distance is a bounded transient).
  ctrl = make_controller(personality=personality)
  vdebt = 0.0
  last = 0.0
  raw = -0.9                                          # armed (raw > SOFT_ONSET_MAX_INSTANT_ACCEL = -1.0)
  for _ in range(500):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    vdebt = max(0.0, vdebt + (out - raw) * DT_MDL)   # accrue only shallower-than-plan deficit
    last = out
  assert vdebt <= _ONSET_VDEBT_BOUND[personality]
  assert last == pytest.approx(raw, abs=1e-3)        # converged to the plan, no permanent offset


@pytest.mark.parametrize("personality", [ECO, SPORT])
def test_convex_onset_no_jerk_snap(personality):
  # A gentle armed bite opens a soft gap; when the plan then deepens past the gentle zone the gap is
  # closed by the FAST hand-back -- still jerk-limited, never a 1-frame snap to the plan. Max realized
  # jerk anywhere on the convex path must not exceed the hand-back ceiling.
  ctrl = make_controller(personality=personality)
  ctrl.smooth_target_accel(0.0, flat_traj(0.0), T_IDXS, should_stop=False)
  prev = 0.0
  worst = 0.0
  for raw in [-0.3] * 10 + [-0.9] * 40:          # gentle onset, then deepen past the gentle zone
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    worst = max(worst, abs(out - prev) / DT_MDL)
    prev = out
  assert worst <= ONSET_HANDBACK_JERK[personality] + _EPS


def test_hard_brake_bypass():
  # Firm (non-crash) hard brake: the DEEPENING RATE is jerk-limited (no raw stock grab), but depth is
  # never reduced (out never deeper than raw) and full depth is reached within a bounded ramp.
  ctrl = make_controller(personality=ECO)
  raw = HARD_BRAKE_TARGET_ACCEL - 0.5    # -2.0
  last = 0.0
  reached = False
  for _ in range(40):
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False)
    assert ctrl.bypassed()
    assert out >= raw - _EPS                                   # never deeper than the plan
    assert out <= last - HARD_BRAKE_ONSET_JERK * DT_MDL + _EPS  # deepening rate capped
    last = out
    if abs(out - raw) < _EPS:
      reached = True
      break
  assert reached                                               # full plan depth reached


def test_hard_brake_onset_jerk_limited_vs_crash():
  # The firm bypass is rate-limited; a true emergency (crash_cnt>0 / FCW) is NOT -> instant full depth.
  firm = make_controller(personality=ECO)
  out_firm = firm.smooth_target_accel(-3.0, flat_traj(-3.0), T_IDXS, should_stop=False)
  assert out_firm == pytest.approx(-HARD_BRAKE_ONSET_JERK * DT_MDL, abs=_EPS)  # first tick capped from last=0
  crash = make_controller(personality=ECO, crash_cnt=3)
  out_crash = crash.smooth_target_accel(-3.0, flat_traj(-3.0), T_IDXS, should_stop=False)
  assert out_crash == pytest.approx(-3.0, abs=_EPS)            # FCW: instant, never rate-limited


def test_should_stop_bypass():
  # should_stop firm brake: rate-limited onset, never deeper than plan, reaches full depth.
  ctrl = make_controller(personality=ECO)
  last = 0.0
  reached = False
  for _ in range(40):
    out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=True)
    assert ctrl.bypassed()
    assert out >= -1.0 - _EPS
    last = out
    if abs(out + 1.0) < _EPS:
      reached = True
      break
  assert reached


def test_disabled_hard_brake_is_instant_stock():
  # off == stock: a disabled controller must pass the hard brake straight through (no rate cap).
  ctrl = make_controller(enabled=False, personality=ECO)
  out = ctrl.smooth_target_accel(-3.0, flat_traj(-3.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-3.0, abs=_EPS)


def test_fcw_crash_cnt_bypass():
  ctrl = make_controller(personality=ECO, crash_cnt=3)
  out = ctrl.smooth_target_accel(-1.0, flat_traj(-1.0), T_IDXS, should_stop=False)
  assert out == pytest.approx(-1.0, abs=_EPS)
  assert ctrl.bypassed()


def test_blended_never_softens_brake():
  # Blended/e2e (stock_brake): the brake is NEVER softened -> output is never weaker than the plan and
  # the convex soft-onset never engages. (It may still anticipate via the never-weaker front-load.)
  ctrl = make_controller(personality=ECO)
  for raw in [0.0, -0.3, -0.6, -0.9, -1.0, -1.0, -1.0]:
    out = ctrl.smooth_target_accel(raw, flat_traj(raw), T_IDXS, should_stop=False, stock_brake=True)
    assert out <= raw + _EPS                 # never weaker than the plan (no softening)
    assert not ctrl._soft_active             # convex soft-onset disabled in blended


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
