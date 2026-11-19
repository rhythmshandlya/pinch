#!/usr/bin/env bash
# Install Pinch's firmware on the Radxa ZERO 3W (Radxa OS, Debian). Run from the project folder:
#     sudo bash deploy/install.sh
# Before this: enable the overlays in `sudo rsetup` (hardware/ASSEMBLY.md stage 1) and reboot.
set -euo pipefail
DEST=/opt/pinch

echo "== packages"
apt-get update
apt-get install -y python3-venv python3-libgpiod alsa-utils i2c-tools      # (not python3-opencv: Debian 12 has 4.6, without cv2.aruco.ArucoDetector)

echo "== code -> $DEST"
mkdir -p "$DEST"
cp -r crab pinch robot brain policies sim "$DEST"/
python3 -m venv --system-site-packages "$DEST/venv"
"$DEST/venv/bin/pip" install --quiet numpy pyserial smbus2 pillow mujoco "opencv-python-headless>=4.8"   # aarch64 wheels; ArucoDetector needs >= 4.7

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
i2cdetect -y 3 || true          # expect 0x28 (IMU), 0x40 (battery), 0x5A (touch), 0x29/0x30 (cliff)
i2cdetect -y 4 || true          # expect 0x3C (face screen)
ls /sys/class/pwm/ || true      # expect six pwmchipN (arm servos)
aplay -l | grep -i crab || echo "  ! crab-audio card not found: install hardware/radxa/crab-audio.dts in rsetup"

echo "== service"
cp deploy/pinch.service /etc/systemd/system/pinch.service
systemctl daemon-reload
systemctl enable pinch.service
echo "Done. Calibrate first (python robot/calibrate.py, then --check-directions, then --trim-arms),"
echo "then: sudo systemctl start pinch    (logs: journalctl -u pinch -f;  status: curl localhost:8080/status)"
