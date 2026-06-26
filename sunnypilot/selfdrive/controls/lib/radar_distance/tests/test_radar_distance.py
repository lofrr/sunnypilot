"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.radar_distance.radar_distance import \
  RadarDistanceController, HOLD_MAX_FRAMES, FCW_PROB_CAP, LOW_SPEED_PASSTHROUGH_V

COMFORT_BRAKE = 2.5


class FakeParams:
  def __init__(self, store=None):
    self.store = dict(store or {})

  def get_bool(self, key):
    return bool(self.store.get(key, False))


def lead(status=True, dRel=40.0, vRel=-2.0, vLead=18.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.95):
  return SimpleNamespace(status=status, dRel=dRel, yRel=0.0, vRel=vRel, vLead=vLead, vLeadK=vLead,
                         aLeadK=aLeadK, aLeadTau=aLeadTau, modelProb=modelProb)


def rs(one, two=None):
  return SimpleNamespace(leadOne=one, leadTwo=two or lead(status=False, dRel=0.0, modelProb=0.0))


def obstacle(ld):
  return ld.dRel + ld.vLead ** 2 / (2 * COMFORT_BRAKE)


def ctrl(enabled=True, vlead_damp=False):
  c = RadarDistanceController(CP=SimpleNamespace(), params=FakeParams({'RadarDistance': enabled}))
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 10.0  # default above the gate so hold-logic tests exercise the flicker-hold
  c._vlead_damp_enabled = vlead_damp         # speed-damp (B) is gated off in production; opt in per-test
  return c


def test_disabled_is_identity():
  c = ctrl(enabled=False)
  r = rs(lead())
  assert c.smooth_radarstate(r) is r  # byte-stock passthrough


def test_valid_lead_passthrough():
  c = ctrl()
  one = lead(dRel=40.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one


def test_holds_after_sustained_dropout():
  c = ctrl()
  for _ in range(3):  # sustain (>= SUSTAIN_FRAMES)
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-4.0, vLead=16.0)))
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  held = out.leadOne
  assert held.status is True
  assert held.dRel < 30.0          # dead-reckoned closer
  assert held.dRel == pytest.approx(30.0 - 4.0 * 0.05, abs=1e-6)


def test_low_speed_override_lead_passthrough():
  # radard low_speed_override emits a real closest-track lead with modelProb=0.0. It must be honored as a
  # real lead (passthrough), NOT rejected and replaced by a stale farther held lead (would under-brake at
  # stop-and-go and stop too close).
  c = ctrl()
  one = lead(status=True, dRel=2.5, vRel=0.0, vLead=0.0, modelProb=0.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one                         # passed straight through, not substituted


def test_low_speed_override_lead_arms_hold():
  # a sustained prob=0 real lead should arm the hold like any real lead
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(status=True, dRel=3.0, vRel=-0.5, vLead=1.0, modelProb=0.0)))
  held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert held.status is True                         # armed off the prob=0 lead, holds through dropout


def test_low_speed_returns_raw_object():
  # Stop/creep regime: ENABLED returns the EXACT raw radarstate object (byte-identical to OFF), so the
  # lead the MPC sees -- and thus the stop distance -- is stock. This is the core stop-neutrality guarantee.
  c = ctrl()
  c._v_ego = LOW_SPEED_PASSTHROUGH_V - 0.1
  r = rs(lead(status=True, dRel=6.0, vRel=0.0, vLead=0.0))
  assert c.smooth_radarstate(r) is r                 # object identity == stock


def test_low_speed_passthrough_but_hold_warmed_for_highway():
  # At low speed the raw radarstate is returned, but the hold is still stepped (state kept warm) so the
  # flicker-hold engages the moment speed rises above the gate.
  c = ctrl()
  for _ in range(3):                                 # sustain a real lead while in the low-speed regime
    c._v_ego = LOW_SPEED_PASSTHROUGH_V - 0.1
    r = rs(lead(dRel=30.0, vRel=-4.0, vLead=16.0))
    assert c.smooth_radarstate(r) is r               # returned object stays raw at low speed
  c._v_ego = LOW_SPEED_PASSTHROUGH_V + 10.0          # rise above the gate -> dropout now held (proxy, not raw)
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is True


def test_vlead_lags_rise_instant_fall():
  c = ctrl(vlead_damp=True)                                     # speed-damp (B) under test; gated off in production
  c.smooth_radarstate(rs(lead(dRel=30.0, vLead=15.0)))           # seed at 15
  rising = c.smooth_radarstate(rs(lead(dRel=30.0, vLead=25.0))).leadOne
  assert 15.0 <= rising.vLead < 25.0                            # rise lagged (<= real -> never faster than real)
  falling = c.smooth_radarstate(rs(lead(dRel=30.0, vLead=8.0))).leadOne
  assert falling.vLead == pytest.approx(8.0, abs=1e-6)          # slow-down instant


def test_vlead_resets_on_track_switch_no_phantom_slow():
  # the old bug: a slow lead's filtered speed carried across a switch to a fast farther track, reporting it
  # near-stopped. A dRel jump (track switch) now resets the filter -> the new track's real speed is reported.
  c = ctrl(vlead_damp=True)                                     # speed-damp (B) under test; gated off in production
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=12.0, vLead=0.5)))         # slow close lead
  switched = c.smooth_radarstate(rs(lead(dRel=80.0, vLead=18.0))).leadOne  # different, far, fast track
  assert switched.vLead == pytest.approx(18.0, abs=1e-6)        # real speed, not the stale ~0.5


def test_vlead_damp_gated_off_reports_real_speed():
  # Production default (VLEAD_DAMP_ENABLED off): a speeding-up lead is NOT lagged -> real vLead passes through
  # (flicker-hold A only). This is the on-by-default behavior; B is opt-in pending on-road proof.
  c = ctrl()                                                    # vlead_damp defaults off (production)
  c.smooth_radarstate(rs(lead(dRel=30.0, vLead=15.0)))
  rising = c.smooth_radarstate(rs(lead(dRel=30.0, vLead=25.0))).leadOne
  assert rising.vLead == pytest.approx(25.0, abs=1e-6)          # no damp -> real speed


def test_obstacle_monotone_during_hold():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, vRel=-4.0, vLead=16.0)))
  last_obs = obstacle(lead(dRel=30.0, vLead=16.0))
  prev = last_obs
  for _ in range(HOLD_MAX_FRAMES):
    held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
    if not held.status:
      break
    o = obstacle(held)
    assert o <= last_obs + 1e-6     # never farther than the last real obstacle (brakes >= last real)
    assert o <= prev + 1e-6         # monotonically non-increasing -> brakes more over the hold
    prev = o


def test_releases_after_hold_cap():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0)))
  statuses = [c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne.status
              for _ in range(HOLD_MAX_FRAMES + 3)]
  assert all(statuses[:HOLD_MAX_FRAMES])        # held through the cap
  assert statuses[HOLD_MAX_FRAMES] is False     # released after


def test_no_hold_without_sustained_lead():
  c = ctrl()
  c.smooth_radarstate(rs(lead(dRel=30.0)))       # single valid frame (< SUSTAIN_FRAMES)
  out = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))
  assert out.leadOne.status is False             # not armed -> no hold


def test_flicker_does_not_reset_wall_clock():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0)))
  # 1/0/1/0 flicker: lone valid frames must NOT reset the wall-clock (sustained < SUSTAIN_FRAMES)
  for _ in range(4):
    c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))  # dropout
    c.smooth_radarstate(rs(lead(dRel=31.0)))                              # lone valid
  assert c._one._since_real > 0                  # wall-clock kept climbing through the flicker


def test_fcw_prob_capped_and_aleadk_not_positive():
  c = ctrl()
  for _ in range(3):
    c.smooth_radarstate(rs(lead(dRel=30.0, aLeadK=1.0, modelProb=0.99)))
  held = c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0))).leadOne
  assert held.modelProb <= FCW_PROB_CAP          # no false FCW from a held phantom
  assert held.aLeadK <= 0.0                       # never project the held lead as accelerating
