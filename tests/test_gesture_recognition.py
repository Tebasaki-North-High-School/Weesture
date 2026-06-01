import numpy as np
from weesture.gesture_recognition import (
    fastdtw,
    resample_sequence,
    GestureRecognizer,
)


def test_fastdtw():
    # Simple case: identical sequences
    x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    assert fastdtw(x, y) == 0

    # Slightly different
    y2 = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    dist = fastdtw(x, y2)
    assert dist >= 0


def test_resample_sequence():
    seq = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
    resampled = resample_sequence(seq, num_samples=5)
    assert len(resampled) == 5
    assert np.allclose(resampled[0], [0, 0, 0])
    assert np.allclose(resampled[-1], [10, 10, 10])
    assert np.allclose(resampled[2], [5, 5, 5])


def test_preprocess():
    recognizer = GestureRecognizer(patterns_dir="non_existent")
    # Test with a simple sequence
    seq = np.array(
        [[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]], dtype=np.float64
    )
    processed = recognizer.preprocess(seq)

    # Now returns 6 features for 3D input (Position + Gradient)
    assert processed.shape == (recognizer.num_samples, 6)
    # Check centering of orientation (first 3 elements should be 0)
    assert np.allclose(processed[0, :3], [0, 0, 0])


def test_parse_log(tmp_path):
    log_dir = tmp_path / "patterns"
    log_dir.mkdir()
    log_file = log_dir / "test_gesture.log"
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


def test_recognize_simple():
    recognizer = GestureRecognizer(patterns_dir="non_existent", threshold=0.5)

    # Create a synthetic pattern: a simple linear move on X axis
    pattern = np.zeros((40, 3))
    pattern[:, 0] = np.linspace(0, 10, 40)

    recognizer.patterns["line"] = [recognizer.preprocess(pattern)]

    # Recognize the same pattern (should match)
    gesture, score = recognizer.recognize(list(map(tuple, pattern)))
    assert gesture == "line"
    assert score < 0.1

    # Create a completely different pattern: constant move on Y axis
    other_pattern = np.zeros((40, 3))
    other_pattern[:, 1] = np.linspace(0, 10, 40)

    gesture, score = recognizer.recognize(list(map(tuple, other_pattern)))
    # It should either not match "line" or have a high score
    assert gesture is None or score > 0.3
