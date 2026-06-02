import time
import os
import sys
import math
from weeee import Wiimote, buttons, ImuFusion


class GestureRecorder:
    def __init__(self, patterns_dir: str = "patterns"):
        self.patterns_dir = patterns_dir
        if not os.path.exists(self.patterns_dir):
            os.makedirs(self.patterns_dir)
            print(f"Created directory: {self.patterns_dir}")

        try:
            self.wiimote = Wiimote()
            print("Connected to Wiimote.")
        except Exception as e:
            print(f"Error connecting to Wiimote: {e}")
            exit(1)

        self.fusion = ImuFusion()

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

    def record(self, gesture_name: str) -> None:
        print(f"\nReady to record gesture: '{gesture_name}'")

        while True:
            # Find the next available filename
            i = 1
            while True:
                filename = os.path.join(self.patterns_dir, f"{gesture_name}_{i}.log")
                if not os.path.exists(filename):
                    break
                i += 1

            print(
                f"\n[Take {i}] Hold 'B' button to start recording. Release 'B' to stop and save."
            )
            print("Press 'Ctrl+C' to finish all recordings.")

            try:
                recording = False
                data_lines: list[str] = []

                # Capture the initial calibration/header info like the original logs
                header = [
                    "Connected to Wiimote.",
                    "MotionPlus activated.",
                    "Calibration complete placeholder",  # For consistency
                    "================================================================================",
                    "Wiimote Console Debugger - Orientation in degrees",
                    "  H = Home button pressed (yaw reset)",
                    "================================================================================",
                ]

                while True:
                    current_time = time.time()
                    dt = min(current_time - self.last_time, 0.1)
                    self.last_time = current_time

                    self.wiimote.update()
                    ax, ay, az = self.wiimote.gforce

                    raw = {"roll": 0, "pitch": 0, "yaw": 0}
                    if self.has_gyro:
                        raw = self.wiimote.gyro_raw
                        self.fusion.update(
                            ax,
                            ay,
                            az,
                            gyro=raw,
                            dt=dt,
                            gyro_slow=self.wiimote.gyro_slow,
                        )
                    else:
                        self.fusion.update(ax, ay, az, dt=dt)

                    if self.wiimote.is_pressed(buttons.BUTTON_HOME):
                        self.fusion.reset_yaw()

                    is_b_pressed = self.wiimote.is_pressed(buttons.BUTTON_B)

                    if is_b_pressed and not recording:
                        recording = True
                        data_lines = []
                        print(f"Recording '{gesture_name}' take {i}...")

                    if recording:
                        y, p, r = (
                            self.fusion.yaw_deg,
                            self.fusion.pitch_deg,
                            self.fusion.roll_deg,
                        )
                        acc_mag = math.sqrt(ax**2 + ay**2 + az**2)
                        # Format matching the original logs
                        line = (
                            f"yaw={y:+7.1f}°  pitch={p:+7.1f}°  roll={r:+7.1f}°  "
                            f"raw=({raw['roll']:5d},{raw['pitch']:5d},{raw['yaw']:5d})  "
                            f"acc=({ax:+.3f},{ay:+.3f},{az:+.3f})  "
                            f"|g|={acc_mag:.3f}  dt={int(dt * 1000)}ms    F"
                        )
                        data_lines.append(line)

                        if not is_b_pressed:
                            recording = False
                            print(f"Finished recording {len(data_lines)} samples.")

                            if len(data_lines) < 10:
                                print("Recording too short, discarding.")
                                break

                            with open(filename, "w", encoding="utf-8") as f:
                                for h in header:
                                    f.write(h + "\n")
                                for d in data_lines:
                                    f.write(d + "\n")

                            print(f"Saved to: {filename}")
                            print("Wait a moment before next take...")
                            time.sleep(1.0)
                            break  # Go to next take

                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("\nRecording session ended.")
                return


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: weesture-record <gesture_name>")
        sys.exit(1)

    gesture_name = sys.argv[1]
    recorder = GestureRecorder()
    recorder.record(gesture_name)


if __name__ == "__main__":
    main()
