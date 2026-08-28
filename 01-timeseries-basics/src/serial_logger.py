import serial, csv
from datetime import datetime

PORT = "COM3"
BAUD = 115200
OUT  = "data/raw/heat_log.csv"

ser = serial.Serial(PORT, BAUD, timeout=2)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "temp_c"])

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if ":" in line:
                line = line.split(":")[-1].strip()

            try:
                temp = float(line)
            except ValueError:
                continue

            ts = datetime.now().isoformat(timespec="milliseconds")
            w.writerow([ts, temp])
            f.flush()
            print(ts, temp)
    except KeyboardInterrupt:
        ser.close()
        print("stopped")