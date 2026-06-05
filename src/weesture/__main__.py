import time
from weeee import Wiimote, buttons, ImuFusion
from .gesture_recognition import GestureRecognizer


class WiimoteGestureApp:
    current_sequence: list[tuple[float, ...]]
    has_gyro: bool
    recording: bool
    last_time: float

    def __init__(self) -> None:
        try:
            self.wiimote = Wiimote()
            print("Connected to Wiimote.")
        except Exception as e:
            print(f"Error connecting to Wiimote: {e}")
            exit(1)

        self.fusion = ImuFusion()
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

        self.last_time = time.time()
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
        self.fusion.calibrate_gyro(samples)
        print("\nCalibration complete.")

    def run(self) -> None:
        print("=" * 80)
        print("Gesture Recognizer")
        print("  Hold 'B' button to perform a gesture.")
        print("  Home button to reset yaw.")
        print("=" * 80)

        while True:
            current_time = time.time()
            dt = min(current_time - self.last_time, 0.1)
            self.last_time = current_time

            self.wiimote.update()
            ax, ay, az = self.wiimote.gforce

            if self.has_gyro:
                self.fusion.update(
                    ax,
                    ay,
                    az,
                    gyro=self.wiimote.gyro_raw,
                    dt=dt,
                    gyro_slow=self.wiimote.gyro_slow,
                )
            else:
                self.fusion.update(ax, ay, az, dt=dt)

            if self.wiimote.is_pressed(buttons.BUTTON_HOME):
                self.fusion.reset_yaw()

            # Gesture detection logic: Use B button to trigger recording
            is_b_pressed = self.wiimote.is_pressed(buttons.BUTTON_B)

            if is_b_pressed and not self.recording:
                self.recording = True
                self.current_sequence = []
                print("Recording gesture...")

            if self.recording:
                self.current_sequence.append(
                    (
                        self.fusion.yaw_deg,
                        self.fusion.pitch_deg,
                        self.fusion.roll_deg,
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
