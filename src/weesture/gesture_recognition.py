from __future__ import annotations
import os
import re
import numpy as np
from numba import njit
from scipy.signal import savgol_filter
from typing import Optional, cast
from numpy.typing import NDArray
from operator import itemgetter


@njit(cache=True)  # type: ignore[untyped-decorator]
def fastdtw(  # type: ignore[misc]
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    radius: int = 10,
) -> float:
    len_x, len_y = x.shape[0], y.shape[0]
    previous_row = np.full(len_y + 1, np.inf, dtype=np.float64)
    current_row = np.full(len_y + 1, np.inf, dtype=np.float64)
    previous_row[0] = 0.0

    for i in range(1, len_x + 1):
        start = max(1, i - radius)
        end = min(len_y + 1, i + radius + 1)
        current_row.fill(np.inf)

        diff = y[start - 1 : end - 1] - x[i - 1]
        costs = np.sqrt(np.sum(diff * diff, axis=1))

        for j in range(start, end):
            idx = j - start
            cost = costs[idx]
            v1 = previous_row[j]
            v2 = current_row[j - 1]
            v3 = previous_row[j - 1]

            if v1 < v2:
                if v1 < v3:
                    min_prev = v1
                else:
                    min_prev = v3
            else:
                if v2 < v3:
                    min_prev = v2
                else:
                    min_prev = v3
            current_row[j] = cost + min_prev

        previous_row, current_row = current_row, previous_row

    return float(previous_row[len_y])


def resample_sequence(
    sequence: NDArray[np.float64], num_samples: int = 50
) -> NDArray[np.float64]:
    """
    Resample a sequence to a fixed number of samples using linear interpolation.
    """
    if len(sequence) < 2:
        if len(sequence) == 1:
            return np.repeat(sequence, num_samples, axis=0)
        return np.zeros((num_samples, 3), dtype=np.float64)

    old_indices: NDArray[np.float64] = np.linspace(0.0, 1.0, len(sequence))
    new_indices: NDArray[np.float64] = np.linspace(0.0, 1.0, num_samples)

    resampled: NDArray[np.float64] = np.zeros(
        (num_samples, sequence.shape[1]), dtype=np.float64
    )
    for i in range(sequence.shape[1]):
        target_seq: NDArray[np.float64] = sequence[:, i]
        resampled_result: NDArray[np.float64] = cast(
            NDArray[np.float64], np.interp(new_indices, old_indices, target_seq)
        )
        resampled[:, i] = resampled_result

    return resampled


class GestureRecognizer:
    patterns: dict[str, list[NDArray[np.float64]]]
    threshold: float
    num_samples: int

    def __init__(self, patterns_dir: str = "patterns", threshold: float = 0.8) -> None:
        self.patterns = {}
        self.threshold = threshold
        self.num_samples = 50  # Standard length for comparison
        self.load_patterns(patterns_dir)

    def load_patterns(self, patterns_dir: str) -> None:
        if not os.path.exists(patterns_dir):
            print(f"Warning: Patterns directory {patterns_dir} not found.")
            return

        for filename in os.listdir(patterns_dir):
            if filename.endswith(".log"):
                # Extract gesture name (e.g., 'circle' from 'circle.log' or 'circle_1.log')
                name_parts = re.split(r"[_.]", filename)
                if not name_parts:
                    continue
                name = name_parts[0]
                path = os.path.join(patterns_dir, filename)
                pattern = self.parse_log(path)
                if pattern.size > 0:
                    if name not in self.patterns:
                        self.patterns[name] = []
                    self.patterns[name].append(pattern)
                    print(
                        f"Loaded pattern: {name} from {filename} ({len(pattern)} samples)"
                    )

    def parse_log(self, path: str) -> NDArray[np.float64]:
        pattern_data: list[list[float]] = []
        # Regex to extract yaw, pitch, roll and acc
        line_re = re.compile(
            r"yaw=\s*([+-]?\d+\.\d+)°\s+pitch=\s*([+-]?\d+\.\d+)°\s+roll=\s*([+-]?\d+\.\d+)°.*acc=\(([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)"
        )

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    match = line_re.search(line)
                    if match:
                        m_groups = cast(
                            tuple[str, str, str, str, str, str], match.groups()
                        )
                        v1 = float(m_groups[0])
                        v2 = float(m_groups[1])
                        v3 = float(m_groups[2])
                        v4 = float(m_groups[3])
                        v5 = float(m_groups[4])
                        v6 = float(m_groups[5])
                        pattern_data.append([v1, v2, v3, v4, v5, v6])
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return np.zeros((0, 3), dtype=np.float64)

        if not pattern_data:
            return np.zeros((0, 3), dtype=np.float64)

        arr: NDArray[np.float64] = np.array(pattern_data, dtype=np.float64)
        return self.preprocess(arr)

    def preprocess(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize and resample the sequence.
        Supports 3D (orientation) or 6D (orientation + acceleration).
        """
        if len(sequence) == 0:
            cols = sequence.shape[1] if sequence.ndim > 1 else 3
            return np.zeros((self.num_samples, cols), dtype=np.float64)

        current_seq: NDArray[np.float64] = sequence.copy()
        is_6d: bool = current_seq.shape[1] == 6

        # 1. Angle unwrapping
        sequence_rad: NDArray[np.float64] = np.radians(current_seq[:, :3])
        sequence_unwrapped: NDArray[np.float64] = np.unwrap(sequence_rad, axis=0)
        current_seq[:, :3] = np.degrees(sequence_unwrapped)

        # 2. Smoothing (Savitzky-Golay)
        if len(current_seq) > 11:
            current_seq = savgol_filter(
                current_seq, window_length=7, polyorder=2, axis=0
            )
        elif len(current_seq) > 5:
            current_seq = savgol_filter(
                current_seq, window_length=5, polyorder=2, axis=0
            )

        # 3. Trim sequence (Endpoint Detection)
        if len(current_seq) > 10:
            grad: NDArray[np.float64] = np.gradient(current_seq[:, :3], axis=0)
            energy: NDArray[np.float64] = np.sum(np.abs(grad), axis=1)
            max_energy: float = float(np.max(energy))
            threshold: float = max_energy * 0.1
            active_indices: NDArray[np.intp] = np.where(energy > threshold)[0]
            if active_indices.size > 0:
                start: int = int(active_indices[0])
                end: int = int(active_indices[-1])
                start = max(0, start - 2)
                end = min(len(current_seq), end + 2)
                current_seq = current_seq[start:end]

        # 4. Resample
        resampled_seq: NDArray[np.float64] = resample_sequence(
            current_seq, self.num_samples
        )

        # 5. Feature Engineering
        orientation: NDArray[np.float64] = resampled_seq[:, :3] - resampled_seq[0, :3]
        ranges: NDArray[np.float64] = np.max(orientation, axis=0) - np.min(
            orientation, axis=0
        )
        max_range: float = float(np.max(ranges))
        if max_range > 1.0:
            orientation = orientation / max_range

        orient_grad: NDArray[np.float64] = np.gradient(orientation, axis=0)
        grad_max: float = float(np.max(np.abs(orient_grad)))
        if grad_max > 0.01:
            orient_grad = orient_grad / grad_max

        result: NDArray[np.float64]
        if is_6d:
            accel: NDArray[np.float64] = resampled_seq[:, 3:]
            accel_mean: NDArray[np.float64] = np.mean(accel, axis=0)
            accel = accel - accel_mean
            acc_std: float = float(np.std(accel))
            if acc_std > 0.05:
                accel = accel / acc_std

            result = np.hstack([orientation, orient_grad * 0.5, accel * 0.5])
        else:
            result = np.hstack([orientation, orient_grad * 0.5])

        return result

    def recognize(
        self, sequence: list[tuple[float, ...]]
    ) -> tuple[Optional[str], float]:
        if not sequence or len(sequence) < 5:
            return None, float("inf")

        seq_arr: NDArray[np.float64] = np.array(sequence, dtype=np.float64)
        processed_sequence: NDArray[np.float64] = self.preprocess(seq_arr)

        results: list[tuple[str, float]] = []

        for name, pattern_list in self.patterns.items():
            best_dist_for_gesture: float = float("inf")
            for pattern in pattern_list:
                dist = fastdtw(processed_sequence, pattern)
                normalized_dist: float = dist / (2 * self.num_samples)

                if normalized_dist < best_dist_for_gesture:
                    best_dist_for_gesture = normalized_dist

            results.append((name, best_dist_for_gesture))

        if not results:
            return None, float("inf")

        results.sort(key=itemgetter(1))
        best_match, min_dist = results[0]

        print(f"Candidates: {', '.join([f'{n}: {d:.2f}' for n, d in results[:3]])}")

        if min_dist > self.threshold:
            return None, min_dist

        return best_match, min_dist


if __name__ == "__main__":
    recognizer = GestureRecognizer()
    for name, patterns in recognizer.patterns.items():
        print(f"Gesture {name}: {len(patterns)} variations")
