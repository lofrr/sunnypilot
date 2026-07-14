"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import pyray as rl


def _radar_track_ranges(live_tracks) -> list[tuple[int, int]]:
  return sorted({(source.startAddress, source.endAddress) for source in live_tracks.trackSources})


def format_radar_tracks_status(live_tracks) -> str:
  ranges = _radar_track_ranges(live_tracks)
  if not ranges:
    return "none"

  range_text = ", ".join(f"0x{start:X}-0x{end:X}" for start, end in ranges)
  track_count = len(live_tracks.points)
  track_label = "track" if track_count == 1 else "tracks"
  return f"{range_text} · {track_count} {track_label}"


def format_radar_tracks_onroad_status(live_tracks) -> str:
  range_text, count_text = format_radar_tracks_onroad_columns(live_tracks)
  if not range_text:
    return count_text

  return "\n".join(f"{radar_range} {track_count}" for radar_range, track_count in zip(range_text.splitlines(), count_text.splitlines(), strict=True))


def format_radar_tracks_onroad_columns(live_tracks) -> tuple[str, str]:
  sources = sorted(live_tracks.trackSources, key=lambda source: (source.startAddress, source.endAddress, source.bus))
  if not sources:
    return "", "none"

  range_text = "\n".join(f"0x{source.startAddress:X}-0x{source.endAddress:X}" for source in sources)
  count_text = "\n".join(str(source.trackCount) for source in sources)
  return range_text, count_text


class RadarTracks:
  def draw_radar_tracks(self, live_tracks, map_to_screen, path_offset_z, track_size=6, screen_offset=(0, 0)):
    for track in live_tracks.points:
      d_rel, y_rel, v_rel, a_rel = track.dRel, track.yRel, track.vRel, track.aRel
      if not (math.isfinite(d_rel) and math.isfinite(y_rel) and math.isfinite(v_rel) and math.isfinite(a_rel)):
        continue

      pt = map_to_screen(d_rel, -y_rel, path_offset_z)
      if pt is None:
        continue

      x, y = pt[0] + screen_offset[0], pt[1] + screen_offset[1]
      rl.draw_circle(int(x), int(y), track_size, rl.Color(0, 255, 64, 255))
