#!/usr/bin/env bash
# Smart Hive - one-command install on Raspberry Pi OS (Bookworm, 64-bit) or Debian/Ubuntu.
#
#   curl -fsSL https://raw.githubusercontent.com/Polluxgnr/smart-hive/main/install/install_pi.sh | bash
#   (or, from a clone:  bash install/install_pi.sh)
#
# What it does: installs system packages, creates a Python environment, downloads the
# trained models, copies config.example.yaml to config.yaml (if missing) and installs two
# services that start at boot: the camera monitor and the dashboard (port 8080).
# Nothing is sent anywhere until you enable ntfy or e-mail in config.yaml.
set -euo pipefail

REPO_URL="${SMART_HIVE_REPO:-https://github.com/Polluxgnr/smart-hive.git}"
DIR="${SMART_HIVE_DIR:-$HOME/smart-hive}"

say() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip libgl1 libglib2.0-0 >/dev/null
if [ -e /usr/bin/rpicam-hello ] || [ -e /usr/bin/libcamera-hello ]; then
  sudo apt-get install -y -qq python3-picamera2 >/dev/null || true
fi

if [ -f "pyproject.toml" ] && grep -q 'name = "smart-hive"' pyproject.toml; then
  DIR="$(pwd)"
elif [ ! -d "$DIR/.git" ]; then
  say "Downloading Smart Hive into $DIR"
  git clone --depth 1 "$REPO_URL" "$DIR"
fi
cd "$DIR"

say "Python environment (this takes a while the first time)"
python3 -m venv --system-site-packages .venv   # system packages give access to picamera2
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e .

say "Models"
.venv/bin/python tools/fetch_weights.py

if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  if [ -e /usr/bin/rpicam-hello ] || [ -e /usr/bin/libcamera-hello ]; then
    sed -i 's/^  source: 0 /  source: picamera /' config.yaml
  fi
  say "Created config.yaml - edit the hive name and zones (see docs/install_en.md)"
fi

say "Services"
USER_NAME="$(id -un)"
for unit in smart-hive smart-hive-dashboard; do
  sed -e "s|@DIR@|$DIR|g" -e "s|@USER@|$USER_NAME|g" "install/$unit.service" | sudo tee "/etc/systemd/system/$unit.service" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now smart-hive-dashboard.service smart-hive.service

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
say "Done"
echo "Dashboard:   http://${IP:-raspberrypi.local}:8080"
echo "Check zones: .venv/bin/smart-hive calibrate --out calibration.jpg"
echo "Logs:        journalctl -u smart-hive -f"
