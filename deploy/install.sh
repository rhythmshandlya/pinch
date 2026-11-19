#!/usr/bin/env bash
# Install Pinch's firmware on the Radxa ZERO 3W (Radxa OS, Debian). Run from the project folder:
#     sudo bash deploy/install.sh
# Before this: enable the overlays in `sudo rsetup` (hardware/ASSEMBLY.md stage 1) and reboot.
set -euo pipefail
DEST=/opt/pinch

echo "== packages"
apt-get update
apt-get install -y python3-venv python3-libgpiod alsa-utils i2c-tools      # (not python3-opencv: Debian 12 has 4.6, without cv2.aruco.ArucoDetector)
# python3-libgpiod is libgpiod 1.6 on Debian 12 (the v1 Python API); robot/hw_real.py speaks both v1 and v2

echo "== code -> $DEST"
mkdir -p "$DEST"
cp -r crab pinch robot brain policies sim "$DEST"/
rm -f "$DEST/robot/calibration.json"                 # calibration lives in /var/lib/pinch, never in the code folder
python3 -m venv --system-site-packages "$DEST/venv"
"$DEST/venv/bin/pip" install --quiet numpy pyserial smbus2 pillow mujoco "opencv-python-headless>=4.8,<5"   # aarch64 wheels; ArucoDetector needs >= 4.7, OpenCV 5 drops CascadeClassifier

echo "== face models (YuNet + SFace from the OpenCV model zoo, ~39 MB; checked against pinch/faces.py SHA256)"
mkdir -p "$DEST/models/faces"
fetch() {   # name, url, sha256
  f="$DEST/models/faces/$1"
  if [ ! -f "$f" ] || ! echo "$3  $f" | sha256sum -c --status; then
    curl -fsSL -o "$f" "$2"
    echo "$3  $f" | sha256sum -c --status || { echo "checksum mismatch: $1"; rm -f "$f"; exit 1; }
  fi
}
fetch face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
fetch face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79

echo "== settings (/etc/pinch.env)"
if [ ! -f /etc/pinch.env ]; then
  cat > /etc/pinch.env <<'EOF'
# Your laptop's Ollama for richer speech understanding (optional: keyword rules work without it)
PINCH_OLLAMA=http://laptop.local:11434
PINCH_LLM=gemma4:e4b
PINCH_MEMORY=/var/lib/pinch/memory.json
PINCH_PORT=8080
EOF
fi
mkdir -p /var/lib/pinch

echo "== checks"
i2cdetect -y 3 || true          # expect 0x28 (IMU), 0x40 (battery), 0x5A (touch)
i2cdetect -y 5 || true          # cliff sensors: 0x29 after a power-up (all five wake there), 0x29-0x33 once the
                                # firmware has run; nothing at all: check I2C5-M0 in rsetup and the wiring
i2cdetect -y 4 || true          # expect 0x3C (face screen)
ls /sys/class/pwm/ || true      # expect six pwmchipN (arm servos)
aplay -l | grep -i crab || echo "  ! crab-audio card not found: install hardware/radxa/crab-audio.dts in rsetup"

echo "== service"
cp deploy/pinch.service /etc/systemd/system/pinch.service
systemctl daemon-reload
systemctl enable pinch.service
echo "Done. Calibrate first, with the firmware's own Python (it saves /var/lib/pinch/calibration.json;"
echo "the firmware refuses to start without it):"
echo "    sudo $DEST/venv/bin/python $DEST/robot/calibrate.py                    (then --check-directions, --trim-arms)"
echo "    sudo $DEST/venv/bin/python $DEST/robot/calibrate_camera.py             (the fisheye camera, with the printed checkerboard)"
echo "then: sudo systemctl start pinch    (logs: journalctl -u pinch -f;  status: curl localhost:8080/status)"
