"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

AccelController is an INPUT shaper for the longitudinal MPC: a per-tier positive-accel ceiling + open-rate
(launch), and an ADD-ONLY, slewed, decel-held follow-gap widen fed to the MPC t_follow. It never shapes the
MPC output, so these tests pin: off == byte-stock; tier ordering; and the t_follow invariants (add-only,
zero below the gate, slew-bounded, decel-hold, capped).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.accel_controller import AccelController
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  ECO, NORMAL, SPORT, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, RISE_RATE_V, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, JERK_SCALE_BP, JERK_SCALE_V, TF_WIDEN_V_BP, TF_WIDEN_BASE_V, \
  TF_WIDEN_TIER, TF_WIDEN_MAX, TF_SLEW_PER_S, TF_DECEL_HOLD_A, AccelerationPersonality

_EPS = 1e-6
_TF_STOCK = 1.45          # a representative stock t_follow (standard personality); the widen is add-only on top
_SLEW_STEP = TF_SLEW_PER_S * DT_MDL


class FakeParams:
  def __init__(self, store=None):
    self.store = dict(store or {})

  def get_bool(self, key):
    return bool(self.store.get(key, False))

  def get(self, key, return_default=False):
    return int(self.store.get(key, 1))

  def put(self, key, val, block=False):
    self.store[key] = val


def make_sm(v_ego=20.0, a_ego=0.0):
  return {'carState': SimpleNamespace(vEgo=v_ego, aEgo=a_ego)}


def make_controller(enabled=True, personality=NORMAL):
  store = {"AccelPersonalityEnabled": enabled, "AccelPersonality": int(personality)}
  ctrl = AccelController(CP=SimpleNamespace(), mpc=SimpleNamespace(), params=FakeParams(store))
  ctrl.update(make_sm())
  return ctrl


def settle(ctrl, v_ego, a_ego=0.0, t_follow=_TF_STOCK, n=400):
  ctrl.update(make_sm(v_ego=v_ego, a_ego=a_ego))
  out = t_follow
  for _ in range(n):
    out = ctrl.get_t_follow(t_follow, v_ego)
  return out


# --- Profiles / off == stock ------------------------------------------------------------------------------

def test_enum_source_parity():
  assert (ECO, NORMAL, SPORT) == (AccelerationPersonality.eco, AccelerationPersonality.normal, AccelerationPersonality.sport)
  assert (PERSONALITY_MIN, PERSONALITY_MAX) == (0, 2)


def test_disabled_forces_normal_and_stock_ceiling():
  ctrl = make_controller(enabled=False, personality=SPORT)
  assert ctrl.personality() == NORMAL
  assert not ctrl.enabled()
  for v in (0.0, 10.0, 25.0, 40.0):
    assert ctrl.get_max_accel(v) == pytest.approx(np.interp(v, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  # off == stock, regardless of v_ego (the speed-dependent open-rate table is bypassed entirely when disabled)
  for v in (0.0, 5.0, 20.0, 40.0):
    assert ctrl.get_rise_rate(v) == STOCK_RISE_RATE


def test_disabled_t_follow_is_identity():
  ctrl = make_controller(enabled=False, personality=SPORT)
  for v in (2.0, 10.0, 20.0, 30.0):
    assert ctrl.get_t_follow(_TF_STOCK, v) == pytest.approx(_TF_STOCK)
    assert ctrl.follow_widen() == 0.0
    assert not ctrl.widen_active()


def test_stock_ceiling_matches_upstream():
  # off must equal upstream get_max_accel table so the feature is byte-stock when disabled.
  assert STOCK_A_CRUISE_MAX_V == [1.6, 1.2, 0.8, 0.6]
  assert A_CRUISE_MAX_BP == [0., 10., 25., 40.]
  assert STOCK_RISE_RATE == 0.05


def test_ceiling_ordering_eco_le_normal_le_sport():
  eco = make_controller(personality=ECO)
  nrm = make_controller(personality=NORMAL)
  spt = make_controller(personality=SPORT)
  for v in (0.0, 10.0, 25.0, 40.0):
    assert eco.get_max_accel(v) <= nrm.get_max_accel(v) + _EPS
    assert nrm.get_max_accel(v) <= spt.get_max_accel(v) + _EPS
  # strictly distinct where the tables diverge (mid speed)
  assert make_controller(personality=ECO).get_max_accel(25.0) < make_controller(personality=SPORT).get_max_accel(25.0)


def test_rise_rate_ordering_and_above_stock():
  # ordering holds at both knots: near a stop (v=0) and at the steady-state speed (v=5)
  assert RISE_RATE_V[ECO][0] < RISE_RATE_V[NORMAL][0] < RISE_RATE_V[SPORT][0]
  assert RISE_RATE_V[ECO][1] < RISE_RATE_V[NORMAL][1] < RISE_RATE_V[SPORT][1]
  # every tier opens the ceiling faster than stock at both knots (fast take-off, never slower than stock)
  assert RISE_RATE_V[ECO][0] > STOCK_RISE_RATE
  assert RISE_RATE_V[ECO][1] > STOCK_RISE_RATE


def test_rise_rate_fast_near_stop_tapers_to_steady_state():
  # Near a stop (v=0) the open-rate must be large/non-binding (NOT the old flat 0.07/0.16/0.24) so launch
  # is never delayed. At/above the v=5 knot it must match the old flat, telemetry-verified steady-state
  # values exactly, so cruise/resume behavior at speed is unchanged.
  for personality, steady_state in ((ECO, 0.07), (NORMAL, 0.16), (SPORT, 0.24)):
    ctrl = make_controller(personality=personality)
    assert ctrl.get_rise_rate(0.0) >= 0.5
    assert ctrl.get_rise_rate(0.0) > steady_state
    assert ctrl.get_rise_rate(5.0) == pytest.approx(steady_state)
    assert ctrl.get_rise_rate(20.0) == pytest.approx(steady_state)  # flat above the v=5 knot


def test_normal_is_distinct_from_stock():
  nrm = make_controller(personality=NORMAL)
  # enabled NORMAL differs from stock (so NORMAL is a real profile, not a stock alias)
  assert nrm.get_max_accel(25.0) != pytest.approx(np.interp(25.0, A_CRUISE_MAX_BP, STOCK_A_CRUISE_MAX_V))
  assert nrm.get_rise_rate(0.0) != STOCK_RISE_RATE
  assert nrm.get_rise_rate(5.0) != STOCK_RISE_RATE


def test_eco_ceiling_matches_lowered_table():
  # ECO's cruise/resume-range ceiling was lowered (launch knot at v=0 unchanged at 1.55).
  eco = make_controller(personality=ECO)
  for v, expected in zip(A_CRUISE_MAX_BP, (1.55, 0.75, 0.35, 0.20), strict=True):
    assert eco.get_max_accel(v) == pytest.approx(expected)


# --- jerk-scale: launch jerk-cost relaxation (MPC input, feeds long_mpc.set_weights) ----------------------

def test_jerk_scale_disabled_is_stock():
  ctrl = make_controller(enabled=False, personality=SPORT)
  for v in (0.0, 2.5, 5.0, 20.0):
    assert ctrl.get_jerk_scale(v) == pytest.approx(1.0)


def test_jerk_scale_relaxed_near_stop_flat_at_speed():
  for personality, relaxed in ((ECO, 0.60), (NORMAL, 0.45), (SPORT, 0.30)):
    ctrl = make_controller(personality=personality)
    assert ctrl.get_jerk_scale(0.0) == pytest.approx(relaxed)
    assert ctrl.get_jerk_scale(5.0) == pytest.approx(1.0)      # back to stock by the v=5 knot
    assert ctrl.get_jerk_scale(20.0) == pytest.approx(1.0)     # flat above the knot


def test_jerk_scale_never_exceeds_stock():
  # relaxation only ever LOWERS the jerk cost (more responsive), never raises it above 1.0 (stock)
  for personality in (ECO, NORMAL, SPORT):
    ctrl = make_controller(personality=personality)
    for v in np.linspace(0.0, 40.0, 20):
      assert ctrl.get_jerk_scale(float(v)) <= 1.0 + _EPS


def test_jerk_scale_tier_ordering_at_stop():
  eco = make_controller(personality=ECO).get_jerk_scale(0.0)
  nrm = make_controller(personality=NORMAL).get_jerk_scale(0.0)
  spt = make_controller(personality=SPORT).get_jerk_scale(0.0)
  assert spt < nrm < eco    # SPORT relaxes the most (most responsive launch), ECO the least


def test_jerk_scale_table_matches_constants():
  for personality in (ECO, NORMAL, SPORT):
    ctrl = make_controller(personality=personality)
    for v in (0.0, 2.0, 5.0, 15.0):
      assert ctrl.get_jerk_scale(v) == pytest.approx(np.interp(v, JERK_SCALE_BP, JERK_SCALE_V[personality]))


# --- t_follow: add-only speed widen -----------------------------------------------------------------------

def test_t_follow_zero_below_gate():
  ctrl = make_controller(personality=NORMAL)
  out = settle(ctrl, v_ego=TF_WIDEN_V_BP[0] - 1.0)     # below the widen onset
  assert out == pytest.approx(_TF_STOCK)
  assert ctrl.follow_widen() == pytest.approx(0.0, abs=1e-6)


def test_t_follow_widens_at_speed():
  ctrl = make_controller(personality=NORMAL)
  out = settle(ctrl, v_ego=TF_WIDEN_V_BP[1] + 5.0)     # flat-widen region, above the band
  expected = _TF_STOCK + TF_WIDEN_BASE_V[1] * TF_WIDEN_TIER[NORMAL]
  assert out == pytest.approx(expected, abs=1e-3)
  assert ctrl.widen_active()


def test_t_follow_add_only_random_walk():
  rng = np.random.default_rng(0)
  for personality in (ECO, NORMAL, SPORT):
    ctrl = make_controller(personality=personality)
    for _ in range(500):
      v = float(rng.uniform(0.0, 40.0))
      a = float(rng.uniform(-3.0, 1.5))
      ctrl.update(make_sm(v_ego=v, a_ego=a))
      out = ctrl.get_t_follow(_TF_STOCK, v)
      assert out >= _TF_STOCK - _EPS                    # never tighter than the stock gap => brake >= stock
      assert ctrl.follow_widen() <= TF_WIDEN_MAX + _EPS  # widen capped


def test_t_follow_tier_ordering_at_speed():
  v = TF_WIDEN_V_BP[1] + 5.0
  eco = settle(make_controller(personality=ECO), v_ego=v)
  nrm = settle(make_controller(personality=NORMAL), v_ego=v)
  spt = settle(make_controller(personality=SPORT), v_ego=v)
  assert eco > nrm > spt                                # ECO roomiest, SPORT tightest


def test_t_follow_slew_bounded():
  ctrl = make_controller(personality=ECO)
  ctrl.update(make_sm(v_ego=35.0, a_ego=0.0))           # big target widen, start from 0
  prev = 0.0
  for _ in range(50):
    ctrl.get_t_follow(_TF_STOCK, 35.0)
    assert ctrl.follow_widen() - prev <= _SLEW_STEP + _EPS   # opens no faster than the slew cap
    prev = ctrl.follow_widen()


def test_t_follow_decel_hold_does_not_shrink_gap():
  ctrl = make_controller(personality=NORMAL)
  settle(ctrl, v_ego=35.0, a_ego=0.0)                   # open the gap fully
  held = ctrl.follow_widen()
  assert held > 0.1
  # now braking (a_ego below the hold threshold) while speed drops into the zero-widen region
  for _ in range(50):
    ctrl.update(make_sm(v_ego=8.0, a_ego=TF_DECEL_HOLD_A - 1.0))
    ctrl.get_t_follow(_TF_STOCK, 8.0)
    assert ctrl.follow_widen() >= held - _EPS           # gap does not ease in while braking
  # once no longer braking, the gap eases back toward the (zero) target
  for _ in range(200):
    ctrl.update(make_sm(v_ego=8.0, a_ego=0.0))
    ctrl.get_t_follow(_TF_STOCK, 8.0)
  assert ctrl.follow_widen() == pytest.approx(0.0, abs=1e-3)


def test_reset_clears_widen():
  ctrl = make_controller(personality=SPORT)
  settle(ctrl, v_ego=35.0)
  assert ctrl.follow_widen() > 0.0
  ctrl.reset()
  assert ctrl.follow_widen() == 0.0


def test_out_of_range_personality_clamps():
  ctrl = make_controller(personality=99)
  assert ctrl.personality() == PERSONALITY_MAX


def test_max_accel_uses_stored_v_ego():
  ctrl = make_controller(personality=SPORT)
  ctrl.update(make_sm(v_ego=0.0))
  assert ctrl.max_accel() == pytest.approx(ctrl.get_max_accel(0.0))
