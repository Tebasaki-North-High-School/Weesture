import time
from weeee import Wiimote, buttons
from .gesture_recognition import GestureRecognizer


class WiimoteGestureApp:
    current_sequence: list[tuple[float, ...]]
    has_gyro: bool
    recording: bool

    def __init__(self) -> None:
        try:
            self.wiimote = Wiimote()
            print("Connected to Wiimote.")
        except Exception as e:
            print(f"Error connecting to Wiimote: {e}")
            exit(1)

        self.recognizer = GestureRecognizer()

        try:
            self.wiimote.require_motion_plus()
            print("MotionPlus activated.")
            self.has_gyro = True
        except Exception as e:
            print(f"MotionPlus error: {e}")
            self.has_gyro = False

        if self.has_gyro:
            self.calibrate_gyro()

        self.recording = False
        self.current_sequence = []

    def calibrate_gyro(self) -> None:
        print("Calibrating gyro... Keep still.", end="", flush=True)
        samples: list[dict[str, int]] = []
        while len(samples) < 100:
            time.sleep(0.01)
            if self.wiimote.update() is not None:
                samples.append(self.wiimote.gyro_raw.copy())
                if len(samples) % 10 == 0:
                    print(".", end="", flush=True)
        assert self.wiimote.fusion is not None
        self.wiimote.fusion.calibrate_gyro(samples)
        self.wiimote.fusion._first_frame = True
        print("\nCalibration complete.")

    def run(self) -> None:
        print("=" * 80)
        print("Gesture Recognizer")
        print("  Hold 'B' button to perform a gesture.")
        print("  Home button to reset yaw.")
        print("=" * 80)

        while True:
            self.wiimote.update()
            ax, ay, az = self.wiimote.gforce

            if self.wiimote.is_pressed(buttons.BUTTON_HOME):
                self.wiimote.reset_yaw()

            # Gesture detection logic: Use B button to trigger recording
            is_b_pressed = self.wiimote.is_pressed(buttons.BUTTON_B)

            if is_b_pressed and not self.recording:
                self.recording = True
                self.current_sequence = []
                print("Recording gesture...")

            if self.recording:
                self.current_sequence.append(
                    (
                        self.wiimote.yaw_deg,
                        self.wiimote.pitch_deg,
                        self.wiimote.roll_deg,
                        ax,
                        ay,
                        az,
                    )
                )

                if not is_b_pressed:
                    self.recording = False
                    print(
                        f"Finished recording ({len(self.current_sequence)} samples). Recognizing..."
                    )
                    gesture, score = self.recognizer.recognize(self.current_sequence)
                    if gesture:
                        print(f">>> DETECTED: {gesture.upper()} (Score: {score:.2f})")
                    else:
                        print(">>> GESTURE NOT RECOGNIZED")

            time.sleep(0.01)


def main() -> None:
    app = WiimoteGestureApp()
    app.run()


if __name__ == "__main__":
    main()
