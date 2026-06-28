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


def lead(status=True, dRel=40.0, vRel=-2.0, vLead=18.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.95, radarTrackId=-1):
  return SimpleNamespace(status=status, dRel=dRel, yRel=0.0, vRel=vRel, vLead=vLead, vLeadK=vLead,
                         aLeadK=aLeadK, aLeadTau=aLeadTau, modelProb=modelProb, radarTrackId=radarTrackId)


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


# --- lead-instability detector (telemetry) -----------------------------------

def test_stability_quiet_on_clean_lead():
  c = ctrl()
  for v in (18.0, 18.2, 17.9, 18.1, 18.0, 17.8):                # steady lead, small noise
    c.smooth_radarstate(rs(lead(dRel=40.0, vLead=v)))
  assert not c.lead_unstable()                                  # range < VLEAD_SPREAD -> stable

def test_stability_flags_bimodal_lead():
  c = ctrl()
  for v in (12.0, 2.0, 12.0, 2.0, 12.0):                        # bouncing between two tracks
    c.smooth_radarstate(rs(lead(dRel=60.0, vLead=v)))
  assert c.lead_unstable()                                      # range 10 m/s > VLEAD_SPREAD -> unstable

def test_stability_resets_on_dropout():
  c = ctrl()
  for v in (12.0, 2.0, 12.0, 2.0, 12.0):
    c.smooth_radarstate(rs(lead(dRel=60.0, vLead=v)))
  assert c.lead_unstable()
  c.smooth_radarstate(rs(lead(status=False, dRel=0.0, modelProb=0.0)))   # lead drops
  assert not c.lead_unstable()                                 # buffer cleared -> stable

def test_stability_runs_even_when_disabled():
  c = ctrl(enabled=False)                                      # telemetry runs regardless of RadarDistance gate
  for v in (12.0, 2.0, 12.0, 2.0, 12.0):
    c.smooth_radarstate(rs(lead(dRel=60.0, vLead=v)))
  assert c.lead_unstable()

def test_stability_flags_trackid_churn():
  c = ctrl()
  for tid in (10, 20, 10, 20, 10, 20, 10, 20, 10, 20):         # steady lead, radarTrackId flipping (follow-hunt)
    c.smooth_radarstate(rs(lead(dRel=44.0, vLead=27.0, radarTrackId=tid)))
  assert c.lead_unstable()

def test_stability_steady_id_quiet():
  c = ctrl()
  for _ in range(10):
    c.smooth_radarstate(rs(lead(dRel=44.0, vLead=27.0, radarTrackId=10)))
  assert not c.lead_unstable()                                 # steady lead + steady id -> stable


# --- lead jitter smoother (B2: anti follow-hunt) -----------------------------

def _churn_feed(c, n=20):
  out = []
  for k in range(n):
    dr = 42.0 if k % 2 == 0 else 46.0                          # steady ~44m lead, dRel jitter
    tid = 10 if k % 2 == 0 else 20                             # radarTrackId churning
    out.append(c.smooth_radarstate(rs(lead(dRel=dr, vLead=27.0, vRel=0.0, radarTrackId=tid))).leadOne.dRel)
  return out

def test_lead_smooth_removes_churn_jitter():
  c = ctrl()
  c._lead_smooth_enabled = True
  tail = _churn_feed(c)[12:]
  assert max(tail) - min(tail) < 3.0                           # raw range is 4.0 -> jitter reduced
  assert all(42.5 < x < 45.5 for x in tail)                   # pulled toward the mean ~44

def test_lead_smooth_off_passthrough():
  c = ctrl()                                                   # smoother off (default)
  tail = _churn_feed(c)[12:]
  assert {round(x, 1) for x in tail} <= {42.0, 46.0}           # raw dRel, no smoothing

def test_lead_smooth_inactive_without_churn():
  c = ctrl()
  c._lead_smooth_enabled = True
  out = None
  for _ in range(12):
    out = c.smooth_radarstate(rs(lead(dRel=44.0, vLead=27.0, radarTrackId=10)))   # steady id -> no churn
  assert out.leadOne.dRel == pytest.approx(44.0, abs=1e-6)     # smoother inactive -> exact dRel


# --- stop-gap bias (smooth farther stop) -------------------------------------

def _biased_ctrl(v_ego=2.0):
  c = ctrl()
  c._stop_gap_bias_enabled = True
  c._v_ego = v_ego
  return c

def _bias(c, dRel, vLead):
  return c._stop_gap_bias(lead(dRel=dRel, vLead=vLead))

def test_stop_bias_pulls_stopped_lead_closer():
  out = _bias(_biased_ctrl(), 8.0, 0.0)
  assert 2.0 <= out.dRel < 8.0                                 # reported closer (farther stop), floored
  assert out.vLead == 0.0 and out.status                       # other fields preserved

def test_stop_bias_monotone_never_farther():
  c = _biased_ctrl()
  for dr in (4.0, 6.0, 8.0, 10.0, 12.0, 20.0):
    assert _bias(c, dr, 0.0).dRel <= dr + 1e-6

def test_stop_bias_min_floor():
  assert _bias(_biased_ctrl(), 2.5, 0.0).dRel == pytest.approx(2.0, abs=1e-6)

def test_stop_bias_off_no_change():
  c = ctrl()
  c._v_ego = 2.0
  ld = lead(dRel=8.0, vLead=0.0)
  assert c._stop_gap_bias(ld) is ld                            # default off -> exact passthrough

def test_stop_bias_moving_lead_no_change():
  ld = lead(dRel=8.0, vLead=5.0)
  assert _biased_ctrl()._stop_gap_bias(ld) is ld

def test_stop_bias_high_speed_no_change():
  ld = lead(dRel=8.0, vLead=0.0)
  assert _biased_ctrl(v_ego=15.0)._stop_gap_bias(ld) is ld

def test_stop_bias_far_lead_no_change():
  ld = lead(dRel=30.0, vLead=0.0)
  assert _biased_ctrl()._stop_gap_bias(ld) is ld               # beyond regime -> no bias

def test_stop_bias_via_smooth_radarstate_low_speed():
  out = _biased_ctrl().smooth_radarstate(rs(lead(dRel=8.0, vLead=0.0, vRel=-2.0)))
  assert out.leadOne.dRel < 8.0                                # biased proxy returned at low speed


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
