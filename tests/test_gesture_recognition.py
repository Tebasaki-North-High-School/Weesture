import pathlib
import time
import numpy as np
from typing import cast
from weesture.gesture_recognition import (
    fastdtw,
    resample_sequence,
    GestureRecognizer,
)


def test_fastdtw() -> None:
    # Simple case: identical sequences
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(((0, 0, 0), (1, 1, 1), (2, 2, 2)), dtype=np.float64)
    y: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(((0, 0, 0), (1, 1, 1), (2, 2, 2)), dtype=np.float64)
    assert fastdtw(x, y) == 0.0

    # Slightly different
    y2: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(((0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1), (2, 2, 2)), dtype=np.float64)
    dist = fastdtw(x, y2)
    assert dist >= 0.0


def test_resample_sequence() -> None:
    seq: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(((0, 0, 0), (10, 10, 10)), dtype=np.float64)
    resampled = resample_sequence(seq, num_samples=5)
    assert len(resampled) == 5
    
    # Casting rows to NDArray to satisfy strict mypy indexing rules
    r0: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], resampled[0])
    rlast: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], resampled[-1])
    r2: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], resampled[2])
    
    assert np.allclose(r0, [0.0, 0.0, 0.0])
    assert np.allclose(rlast, [10.0, 10.0, 10.0])
    assert np.allclose(r2, [5.0, 5.0, 5.0])


def test_preprocess() -> None:
    recognizer = GestureRecognizer(patterns_dir="non_existent")
    # Test with a simple sequence
    seq: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
        ((0, 0, 0), (1, 2, 3), (2, 4, 6), (3, 6, 9), (4, 8, 12)), dtype=np.float64
    )
    processed = recognizer.preprocess(seq)

    # Now returns 6 features for 3D input (Position + Gradient)
    assert processed.shape == (recognizer.num_samples, 6)
    
    p0: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], processed[0, :3])
    # Check centering of orientation (first 3 elements should be 0)
    assert np.allclose(p0, [0.0, 0.0, 0.0])


def test_parse_log(tmp_path: pathlib.Path) -> None:
    log_dir: pathlib.Path = tmp_path / "patterns"
    log_dir.mkdir()
    log_file: pathlib.Path = log_dir / "test_gesture.log"
    content = """
Connected to Wiimote.
MotionPlus activated.
================================================================================
yaw=   10.0°  pitch=   20.0°  roll=   30.0°  raw=(    0,    0,    0)  acc=(+0.000,+0.000,+1.000)  |g|=1.000  dt=10ms    F
yaw=   11.0°  pitch=   21.0°  roll=   31.0°  raw=(    0,    0,    0)  acc=(+0.000,+0.000,+1.000)  |g|=1.000  dt=10ms    F
"""
    log_file.write_text(content, encoding="utf-8")

    recognizer = GestureRecognizer(patterns_dir=str(log_dir))
    assert "test" in recognizer.patterns
    assert len(recognizer.patterns["test"]) == 1
    # Check if preprocessed
    pattern = recognizer.patterns["test"][0]
    # Returns 9 features for 6D input (Orientation + Gradient + Acceleration)
    assert pattern.shape == (recognizer.num_samples, 9)


def test_recognize_simple() -> None:
    recognizer = GestureRecognizer(patterns_dir="non_existent", threshold=0.5)

    # Create a synthetic pattern: a simple linear move on X axis
    pattern: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros((40, 3), dtype=np.float64)
    vals: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], np.linspace(0, 10, 40))
    pattern[:, 0] = vals

    recognizer.patterns["line"] = [recognizer.preprocess(pattern)]

    # Recognize the same pattern (should match)
    # Convert NDArray to list of tuples for recognize
    sequence: list[tuple[float, ...]] = []
    for i in range(len(pattern)):
        row: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], pattern[i])
        sequence.append((float(row[0]), float(row[1]), float(row[2])))
        
    gesture, score = recognizer.recognize(sequence)
    assert gesture == "line"
    assert score < 0.1

    # Create a completely different pattern: constant move on Y axis
    other_pattern: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros((40, 3), dtype=np.float64)
    other_vals: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], np.linspace(0, 10, 40))
    other_pattern[:, 1] = other_vals

    other_sequence: list[tuple[float, ...]] = []
    for i in range(len(other_pattern)):
        row_other: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(np.ndarray[tuple[int], np.dtype[np.float64]], other_pattern[i])
        other_sequence.append((float(row_other[0]), float(row_other[1]), float(row_other[2])))
        
    gesture, score = recognizer.recognize(other_sequence)
    # It should either not match "line" or have a high score
    assert gesture is None or score > 0.3


def test_recognize_many_patterns_performance() -> None:
    recognizer = GestureRecognizer(patterns_dir="non_existent", threshold=0.5)

    pattern_count = 80
    pattern: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
        (80, 3), dtype=np.float64
    )
    vals: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]], np.linspace(0, 20, 80)
    )
    pattern[:, 0] = vals

    for index in range(pattern_count):
        variant = pattern.copy()
        variant[:, 1] = index * 0.01
        recognizer.patterns[f"line_{index}"] = [recognizer.preprocess(variant)]

    sequence: list[tuple[float, ...]] = []
    for i in range(len(pattern)):
        row: np.ndarray[tuple[int], np.dtype[np.float64]] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]], pattern[i]
        )
        sequence.append((float(row[0]), float(row[1]), float(row[2])))

    started_at = time.perf_counter()
    gesture, score = recognizer.recognize(sequence)
    elapsed = time.perf_counter() - started_at

    assert gesture == "line_0"
    assert score < 0.1
    assert elapsed < 0.5
