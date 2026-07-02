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
  DROPOUT_DREL, STOP_GAP_MIN_DREL, STOP_GAP_VEGO, STOP_GAP_VLEAD, STOP_GAP_REGIME_DREL

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


def test_low_speed_override_lead_passthrough():
  # radard low_speed_override emits a real closest-track lead with modelProb=0.0. It must be honored, not
  # rejected in favor of a stale farther held lead (which would under-brake / stop too close).
  c = ctrl()
  one = lead(status=True, dRel=2.5, vRel=0.0, vLead=0.0, modelProb=0.0)
  out = c.smooth_radarstate(rs(one))
  assert out.leadOne is one


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
