import serial
import csv
import time
import sys
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

PORT = "COM7"
BAUD = 115200

DATA_DIR = Path(
    r"C:\Users\kewin\humanoid_challenge\data"
)


# ============================================================
# GET DEMO NUMBER
# ============================================================

if len(sys.argv) != 2:
    print()
    print("Usage:")
    print("python record_capacitance.py DEMO_NUMBER")
    print()
    print("Example:")
    print("python record_capacitance.py 2")
    print()
    sys.exit(1)


try:
    demo_number = int(sys.argv[1])

except ValueError:
    print("ERROR: Demo number must be an integer.")
    sys.exit(1)


if demo_number < 1 or demo_number > 99:
    print("ERROR: Invalid demo number.")
    sys.exit(1)


# ============================================================
# OUTPUT FOLDER
# ============================================================

demo_name = f"demo_{demo_number:02d}"

demo_dir = DATA_DIR / demo_name

demo_dir.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT = demo_dir / "capacitance.csv"


# ============================================================
# PROTECT AGAINST ACCIDENTAL OVERWRITE
# ============================================================

if OUTPUT.exists():

    print()
    print("====================================")
    print("WARNING")
    print("====================================")
    print()
    print("This file already exists:")
    print(OUTPUT)
    print()
    print("Recording has NOT started.")
    print()
    print(
        "Delete/rename the existing file if you "
        "really want to record this demo again."
    )
    print()

    sys.exit(1)


# ============================================================
# CONNECT TO ARDUINO
# ============================================================

print()
print("Connecting to Arduino...")

try:

    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=1
    )

except serial.SerialException as e:

    print()
    print("ERROR: Could not open Arduino.")
    print(e)
    sys.exit(1)


# Arduino normally resets when serial opens.
time.sleep(3)

ser.reset_input_buffer()


# ============================================================
# START RECORDING
# ============================================================

print()
print("====================================")
print("CAPACITANCE RECORDING")
print("====================================")
print()
print("Demo:")
print(demo_name)
print()
print("Recording to:")
print(OUTPUT)
print()
print("Start Record3D now.")
print()
print("Perform your synchronization touches,")
print("then perform the manipulation task.")
print()
print("Press Ctrl+C when finished.")
print()


start_pc = time.perf_counter()

rows = 0


try:

    with open(
        OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "pc_time_s",
            "arduino_time_ms",
            "filtered"
        ])

        while True:

            line = (
                ser.readline()
                .decode(
                    "utf-8",
                    errors="ignore"
                )
                .strip()
            )

            if not line:
                continue


            parts = line.split(",")


            # Arduino sends:
            #
            # time_ms,
            # filtered,
            # baseline,
            # difference

            if len(parts) != 4:
                continue


            try:

                arduino_ms = int(
                    parts[0]
                )

                filtered = int(
                    parts[1]
                )

            except ValueError:

                continue


            pc_time = (
                time.perf_counter()
                - start_pc
            )


            writer.writerow([
                pc_time,
                arduino_ms,
                filtered
            ])


            # Immediately write data to disk.
            f.flush()

            rows += 1


            if rows % 25 == 0:

                print(
                    f"{demo_name} | "
                    f"time={pc_time:.2f}s | "
                    f"filtered={filtered}"
                )


except KeyboardInterrupt:

    print()
    print("====================================")
    print("RECORDING STOPPED")
    print("====================================")
    print()
    print("Demo:")
    print(demo_name)
    print()
    print("Samples:")
    print(rows)
    print()
    print("Saved:")
    print(OUTPUT)
    print()


finally:

    ser.close()