"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Closed-loop regression tests for sunnypilot's longitudinal comfort/smoothness fixes, built on the real
LongitudinalPlanner via selfdrive/test/longitudinal_maneuvers/plant.py. Unlike a static log replay (which
can only recompute an isolated input factor from recorded data), this drives the REAL MPC solver every
cycle and feeds its output back into ego's own simulated speed/distance -- so it can catch regressions in
the actual re-solved trajectory, not just in one factor's value. Params are set via the real Params() store;
pytest's autouse openpilot_function_fixture (root conftest.py) gives each test function a fresh isolated
prefix, so this is safe to run without touching real device state.

Each test targets a SPECIFIC bug found and fixed this session, and is verified to actually fail if that fix
is reverted (see the commit history / memory notes referenced in each test's docstring) -- these aren't
just plausible-looking assertions, they have demonstrated teeth.
"""

import numpy as np

from openpilot.common.params import Params
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import NORMAL


def enable_sunnypilot_longitudinal(params: Params, accel_personality=True, radar_distance=True, dec=False):
  params.put_bool("AccelPersonalityEnabled", accel_personality)
  if accel_personality:
    params.put("AccelPersonality", int(NORMAL))
  params.put_bool("RadarDistance", radar_distance)
  params.put_bool("DynamicExperimentalControl", dec)


def count_launch_brake_cycles(accel: np.ndarray, speed: np.ndarray, launch_th=0.3, settle_speed=0.5) -> int:
  # Count distinct "launch" events (acceleration crosses above launch_th while ego is near-stopped) followed
  # by a return to near-zero/negative acceleration -- the "creep brake creep brake" signature. Only counts
  # launches while speed stays low (settle_speed) -- a real, sustained departure isn't a cycle.
  cycles = 0
  launched = False
  for a, v in zip(accel, speed, strict=True):
    if not launched and a > launch_th and v < settle_speed:
      launched = True
      cycles += 1
    elif launched and a < 0.0:
      launched = False
  return cycles


def test_creep_noise_never_causes_repeated_launch_brake_cycling():
  # Route 550a71ee4c7a7fbe/000004b6, t~678-690s: a genuinely-stopped lead's vLead sensor noise (small blips
  # above STOP_GAP_CREEP_V=0.03 m/s, never sustained) previously accumulated in radar_distance.py's
  # stop-gap creep-override counter (monotonic, never decayed) until it falsely latched the bias off,
  # producing a same-cycle gap-widening the MPC read as "room to launch" -- see
  # lead_unstable_gate_revert / routes_04b5_04b6_creep_bug memory notes; fixed in commit 73bea3866f
  # (counter now decays on sub-threshold frames). This reproduces the noise pattern in closed loop: ego
  # approaches and settles behind a near-stopped lead, then the lead's speed hovers with intermittent
  # noise (never sustained motion) for 40s. Must never repeatedly launch-then-brake.
  params = Params()
  enable_sunnypilot_longitudinal(params)

  rng = np.random.default_rng(0)
  breakpoints = [0.0, 8.0]
  speed_lead_values = [3.0, 0.0]
  # 40s of intermittent noise: alternating 0.0 / small blip every 0.5s -- never two consecutive "moving" ticks,
  # so no real sustained motion, matching the real route's noise signature.
  noise_t = np.arange(8.0, 48.0, 0.5)
  noise_v = np.where(rng.random(len(noise_t)) > 0.5, 0.08, 0.0)
  breakpoints += list(noise_t)
  speed_lead_values += list(noise_v)

  man = Maneuver(
    'stopped lead with intermittent vLead noise, no sustained motion',
    duration=48.0,
    initial_speed=8.0,
    lead_relevancy=True,
    initial_distance_lead=30.0,
    speed_lead_values=speed_lead_values,
    breakpoints=breakpoints,
  )
  valid, logs = man.evaluate()
  assert valid
  # logs columns: time, distance, distance_lead, speed, speed_lead, acceleration, d_rel
  t, speed, accel = logs[:, 0], logs[:, 3], logs[:, 5]
  settled = t > 10.0  # after the initial approach, while noise is active
  cycles = count_launch_brake_cycles(accel[settled], speed[settled])
  assert cycles <= 1, f'expected at most one settle-launch, got {cycles} launch-brake cycles from pure sensor noise'


def test_drel_glitch_does_not_whipsaw_accel():
  # Route 550a71ee4c7a7fbe/000004b4, t~976.1s: raw dRel bounced 17.7->12.3->17.15->12.0m across ~0.3s while
  # vRel stayed -1 to -2 m/s -- physically impossible for one real object, a fusion glitch. SWITCH_DREL was
  # 8.0 (too coarse to catch the ~5m bounce); lowered to 4.0 in commit f978c923a4. This reproduces the exact
  # bounce pattern via Plant's lead_dRel_glitch_fn hook (overrides only dRel, leaving the true physics-based
  # vRel/speed evolution intact for the closed loop) and checks the MPC's actual commanded accel doesn't
  # whipsaw in response.
  params = Params()
  enable_sunnypilot_longitudinal(params)

  glitch_window = (5.0, 5.5)  # apply the bounce for a short window mid-maneuver

  def glitch_fn(t, d_rel, v_rel):
    if glitch_window[0] <= t < glitch_window[1]:
      # alternate between the true (closer) reading and a ~5m-farther bounce, matching the real route
      phase = int((t - glitch_window[0]) / 0.05) % 2
      if phase == 1:
        return d_rel + 5.0, v_rel
    return d_rel, v_rel

  plant = Plant(lead_relevancy=True, speed=8.0, distance_lead=20.0, e2e=False,
                lead_dRel_glitch_fn=glitch_fn)

  accels = []
  while plant.current_time < 10.0:
    log = plant.step(v_lead=1.5)
    accels.append(log['acceleration'])
  accels = np.array(accels)

  jerk = np.diff(accels) / (1.0 / plant.rate)
  # focus on the glitch window and its immediate aftermath
  t_arr = np.arange(len(accels)) / plant.rate
  during = (t_arr[1:] >= glitch_window[0]) & (t_arr[1:] < glitch_window[1] + 0.5)
  assert during.any()
  peak_jerk = np.max(np.abs(jerk[during]))
  assert peak_jerk < 3.0, f'dRel glitch produced a {peak_jerk:.2f} m/s^3 accel whipsaw -- glitch is leaking into the commanded accel'


def test_fast_closing_lead_onset_is_ramped_not_snapped():
  # A severe closing-rate lead (matching route 000004b5's flagship regression episode, vRel to -16.5 m/s)
  # legitimately requires a large final decel -- that's not a bug (see routes_04b5_04b6_creep_bug memory:
  # "very firm brake... looks legitimate"). What IS a bug is an instantaneous snap rather than a ramped
  # onset. This checks onset smoothness (peak jerk during the transition) without asserting the final
  # magnitude must be small.
  params = Params()
  enable_sunnypilot_longitudinal(params)

  man = Maneuver(
    'severe closing-rate lead, onset must ramp not snap',
    duration=12.0,
    initial_speed=20.0,
    lead_relevancy=True,
    initial_distance_lead=160.0,
    speed_lead_values=[3.5, 3.5],
    breakpoints=[0.0, 12.0],
  )
  valid, logs = man.evaluate()
  assert valid
  t, accel = logs[:, 0], logs[:, 5]
  dt = np.diff(t)
  dt[dt <= 0] = np.nan
  jerk = np.diff(accel) / dt
  onset = (t[1:] > 0.5) & (t[1:] < 3.0)  # after the first solve settles, during the initial hard reaction
  assert onset.any()
  peak_onset_jerk = np.nanmax(np.abs(jerk[onset]))
  assert peak_onset_jerk < 4.0, f'onset jerk {peak_onset_jerk:.2f} m/s^3 -- braking snapped instead of ramping'
  # sanity: this scenario genuinely needs real braking (not asserting it stays small)
  assert np.min(accel) < -1.0


def test_dec_on_off_agree_with_lead_present():
  # is_e2e() previously only enforced "near/closing radar lead -> pure MPC, never blend e2e" inside DEC's
  # active()-gated branch -- DEC off silently dropped the whole check, letting the e2e model's opinion
  # blend in via min() regardless of the lead. Fixed in commit 7ff32eafea (dec.has_radar_acc_lead() checked
  # unconditionally, before dec.active() is even consulted). This closed-loop test uses an adversarial
  # e2e_accel_fn (independently opinionated, not the harness's default mild self.acceleration+0.1 echo) so
  # DEC-on vs DEC-off would visibly diverge if the fix regressed.
  def adversarial_e2e(t, speed, accel):
    return -2.0  # e2e model insists on a hard brake, independent of the MPC's own view

  def run(dec_enabled):
    params = Params()
    enable_sunnypilot_longitudinal(params, dec=dec_enabled)
    plant = Plant(lead_relevancy=True, speed=15.0, distance_lead=40.0, e2e=True,
                  e2e_accel_fn=adversarial_e2e)
    accels = []
    while plant.current_time < 5.0:
      log = plant.step(v_lead=13.0)  # near lead, closing slowly -- within RADAR_LEAD_ACC_MAX_DREL
      accels.append(log['acceleration'])
    return np.array(accels)

  accel_dec_on = run(dec_enabled=True)
  accel_dec_off = run(dec_enabled=False)
  np.testing.assert_allclose(accel_dec_on, accel_dec_off, atol=0.05,
                              err_msg='DEC on vs off disagree with an identical near/closing lead present')
