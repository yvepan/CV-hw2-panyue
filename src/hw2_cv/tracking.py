from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _side(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


class LineCounter:
    def __init__(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        self.a = a
        self.b = b
        self.last_side: dict[int, float] = {}
        self.counted: set[int] = set()

    @property
    def count(self) -> int:
        return len(self.counted)

    def update(self, track_id: int, center: tuple[float, float], frame: int) -> dict[str, Any] | None:
        s = _side(center, self.a, self.b)
        prev = self.last_side.get(track_id)
        self.last_side[track_id] = s
        if prev is None or track_id in self.counted:
            return None
        if prev == 0 or s == 0 or (prev > 0) == (s > 0):
            return None
        self.counted.add(track_id)
        direction = "a_to_b_right" if prev < s else "b_to_a_left"
        return {
            "frame": frame,
            "track_id": track_id,
            "cx": center[0],
            "cy": center[1],
            "direction": direction,
            "total": self.count,
        }


def analyze_id_stability(rows: list[dict[str, Any]], start_frame: int, num_frames: int = 4) -> dict[str, Any]:
    selected_frames = list(range(start_frame, start_frame + num_frames))
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frame = int(row["frame"])
        if frame in selected_frames:
            by_frame[frame].append(row)

    details = []
    kept_total = 0
    lost_total = 0
    jump_total = 0
    for prev_f, cur_f in zip(selected_frames, selected_frames[1:]):
        prev = by_frame.get(prev_f, [])
        cur = by_frame.get(cur_f, [])
        prev_ids = {int(r["track_id"]) for r in prev}
        cur_ids = {int(r["track_id"]) for r in cur}
        kept = sorted(prev_ids & cur_ids)
        lost = sorted(prev_ids - cur_ids)
        new = sorted(cur_ids - prev_ids)
        jumps = []
        unmatched_prev = [r for r in prev if int(r["track_id"]) in lost]
        unmatched_cur = [r for r in cur if int(r["track_id"]) in new]
        for p in unmatched_prev:
            for c in unmatched_cur:
                dist = math.hypot(float(p["cx"]) - float(c["cx"]), float(p["cy"]) - float(c["cy"]))
                if dist < 0.08:
                    jumps.append({"from_id": int(p["track_id"]), "to_id": int(c["track_id"]), "distance": dist})
        kept_total += len(kept)
        lost_total += len(lost)
        jump_total += len(jumps)
        details.append(
            {
                "from_frame": prev_f,
                "to_frame": cur_f,
                "kept_ids": kept,
                "lost_ids": lost,
                "new_ids": new,
                "possible_id_jumps": jumps,
            }
        )

    return {
        "start_frame": start_frame,
        "num_frames": num_frames,
        "frames": selected_frames,
        "tracks_per_frame": {str(f): len(by_frame.get(f, [])) for f in selected_frames},
        "kept_id_links": kept_total,
        "lost_id_links": lost_total,
        "possible_id_jumps": jump_total,
        "details": details,
    }

