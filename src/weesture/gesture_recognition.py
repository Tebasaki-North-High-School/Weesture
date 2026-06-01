import re
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from typing import Optional
from collections.abc import Callable
from numpy.typing import NDArray
from operator import itemgetter


def fastdtw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dist: Callable[..., float] = euclidean,
    radius: int = 10,
) -> np.float64:
    """
    A simple implementation of Dynamic Time Warping with Sakoe-Chiba band.
    """
    len_x, len_y = len(x), len(y)
    dtw_matrix = np.full((len_x + 1, len_y + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, len_x + 1):
        # Apply Sakoe-Chiba band
        start = max(1, i - radius)
        end = min(len_y + 1, i + radius + 1)
        for j in range(start, end):
            cost = dist(x[i - 1], y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[len_x, len_y]


def resample_sequence(sequence: NDArray[np.float64], num_samples: int = 50):
    """
    Resample a sequence to a fixed number of samples using linear interpolation.
    """
    if len(sequence) < 2:
        if len(sequence) == 1:
            return np.repeat(sequence, num_samples, axis=0)
        return np.zeros((num_samples, 3))

    sequence = np.array(sequence)
    old_indices = np.linspace(0, 1, len(sequence))
    new_indices = np.linspace(0, 1, num_samples)

    resampled = np.zeros((num_samples, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        resampled[:, i] = np.interp(new_indices, old_indices, sequence[:, i])

    return resampled


class GestureRecognizer:
    def __init__(self, patterns_dir: str = "patterns", threshold: float = 0.8) -> None:
        self.patterns: dict[str, list[NDArray[np.float64]]] = {}
        self.threshold = threshold
        self.num_samples = 50  # Standard length for comparison
        self.load_patterns(patterns_dir)

    def load_patterns(self, patterns_dir: str):
        import os

        if not os.path.exists(patterns_dir):
            print(f"Warning: Patterns directory {patterns_dir} not found.")
            return

        for filename in os.listdir(patterns_dir):
            if filename.endswith(".log"):
                # Extract gesture name (e.g., 'circle' from 'circle.log' or 'circle_1.log')
                name = re.split(r"[_.]", filename)[0]
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
        pattern: list[list[float]] = []
        # Regex to extract yaw, pitch, roll and acc
        line_re = re.compile(
            r"yaw=\s*([+-]?\d+\.\d+)°\s+pitch=\s*([+-]?\d+\.\d+)°\s+roll=\s*([+-]?\d+\.\d+)°.*acc=\(([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)"
        )

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    match = line_re.search(line)
                    if match:
                        # y, p, r, ax, ay, az
                        values = list(map(float, match.groups()))
                        pattern.append(values)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return np.array([])

        if not pattern:
            return np.array([])

        return self.preprocess(np.array(pattern))

    def preprocess(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize and resample the sequence.
        Supports 3D (orientation) or 6D (orientation + acceleration).
        """
        if len(sequence) == 0:
            return np.zeros(
                (self.num_samples, sequence.shape[1] if sequence.ndim > 1 else 3)
            )

        sequence = np.array(sequence)
        is_6d = sequence.shape[1] == 6

        # 1. Angle unwrapping (only for the first 3 columns: yaw, pitch, roll)
        sequence_rad = np.radians(sequence[:, :3])
        sequence_unwrapped = np.unwrap(sequence_rad, axis=0)
        sequence[:, :3] = np.degrees(sequence_unwrapped)

        # 2. Smoothing (Savitzky-Golay)
        if len(sequence) > 11:
            sequence = savgol_filter(sequence, window_length=7, polyorder=2, axis=0)
        elif len(sequence) > 5:
            sequence = savgol_filter(sequence, window_length=5, polyorder=2, axis=0)

        # 3. Trim sequence (Endpoint Detection)
        # Use variance or gradient of orientation to find start/end of movement
        if len(sequence) > 10:
            energy = np.sum(np.abs(np.gradient(sequence[:, :3], axis=0)), axis=1)
            threshold = np.max(energy) * 0.1
            active_indices = np.where(energy > threshold)[0]
            if active_indices.size > 0:
                start, end = active_indices[0], active_indices[-1]
                # Add some buffer
                start = max(0, start - 2)
                end = min(len(sequence), end + 2)
                sequence = sequence[start:end]

        # 4. Resample to standard length
        sequence = resample_sequence(sequence, self.num_samples)

        # 5. Feature Engineering
        # Orientation: center relative to start
        orientation = sequence[:, :3] - sequence[0, :3]
        ranges = np.max(orientation, axis=0) - np.min(orientation, axis=0)
        max_range = np.max(ranges)
        if max_range > 1.0:
            orientation = orientation / max_range

        # Orientation Gradient (Angular Velocity)
        orient_grad = np.gradient(orientation, axis=0)
        grad_max = np.max(np.abs(orient_grad))
        if grad_max > 0.01:
            orient_grad = orient_grad / grad_max

        if is_6d:
            # Acceleration: standardize (mean=0, std=1) for each trial
            accel = sequence[:, 3:]
            accel = accel - np.mean(accel, axis=0)
            acc_std = np.std(accel)
            if acc_std > 0.05:
                accel = accel / acc_std

            # Combine: Orient (3) + Grad (3) + Accel (3) = 9 features
            sequence = np.hstack([orientation, orient_grad * 0.5, accel * 0.5])
        else:
            # Combine: Orient (3) + Grad (3) = 6 features
            sequence = np.hstack([orientation, orient_grad * 0.5])

        return sequence

    def recognize(
        self, sequence: list[tuple[float, ...]]
    ) -> tuple[Optional[str], float]:
        if not sequence or len(sequence) < 5:
            return None, float("inf")

        processed_sequence: NDArray[np.float64] = self.preprocess(np.array(sequence))

        results: list[tuple[str, np.float64]] = []

        for name, pattern_list in self.patterns.items():
            best_dist_for_gesture = np.float64("inf")
            for pattern in pattern_list:
                # DTW distance on preprocessed sequences
                dist = fastdtw(processed_sequence, pattern)
                # Normalize by total steps in the DTW path (approx 2 * num_samples)
                normalized_dist = dist / (2 * self.num_samples)

                if normalized_dist < best_dist_for_gesture:
                    best_dist_for_gesture = normalized_dist

            results.append((name, best_dist_for_gesture))

        if not results:
            return None, float("inf")

        # Sort by distance
        results.sort(key=itemgetter(1))

        best_match, min_dist = results[0]

        # print candidates for debugging
        print(f"Candidates: {', '.join([f'{n}: {d:.2f}' for n, d in results[:3]])}")

        if min_dist > self.threshold:
            return None, min_dist

        return best_match, min_dist


if __name__ == "__main__":
    # Test loading
    recognizer = GestureRecognizer()
    for name, patterns in recognizer.patterns.items():
        print(f"Gesture {name}: {len(patterns)} variations")
