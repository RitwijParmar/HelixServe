#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:?set PROJECT_ID}
INSTANCE_NAME=${INSTANCE_NAME:?set INSTANCE_NAME}
ZONE=${ZONE:-us-central1-c}
IDLE_MINUTES=${IDLE_MINUTES:-30}
CHECK_PORT=${CHECK_PORT:-8000}
GPU_UTIL_THRESHOLD=${GPU_UTIL_THRESHOLD:-5}
MIN_UPTIME_MINUTES=${MIN_UPTIME_MINUTES:-15}

gcloud compute ssh "$INSTANCE_NAME" --project "$PROJECT_ID" --zone "$ZONE" --command "
set -euo pipefail
sudo mkdir -p /var/lib/helixserve

cat <<'EOF' | sudo tee /usr/local/bin/helixserve_idle_shutdown.sh >/dev/null
#!/usr/bin/env bash
set -euo pipefail

IDLE_MINUTES=30
CHECK_PORT=8000
GPU_UTIL_THRESHOLD=5
MIN_UPTIME_MINUTES=15
STATE_FILE=/var/lib/helixserve/last_active_epoch
DISABLE_FILE=/var/lib/helixserve/disable_idle_shutdown

if [[ -f /etc/default/helixserve-idle ]]; then
  # shellcheck disable=SC1091
  source /etc/default/helixserve-idle
fi

if [[ -f \"\$DISABLE_FILE\" ]]; then
  exit 0
fi

now_epoch=\$(date +%s)
uptime_seconds=\$(cut -d. -f1 /proc/uptime)
if (( uptime_seconds < MIN_UPTIME_MINUTES * 60 )); then
  exit 0
fi

mkdir -p \"\$(dirname \"\$STATE_FILE\")\"
if [[ ! -f \"\$STATE_FILE\" ]]; then
  echo \"\$now_epoch\" > \"\$STATE_FILE\"
fi

has_active_connections=0
if ss -Htan state established \"( sport = :\${CHECK_PORT} )\" | grep -q .; then
  has_active_connections=1
fi

gpu_util=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_util_raw=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
  if [[ \"\$gpu_util_raw\" =~ ^[0-9]+$ ]]; then
    gpu_util=\$gpu_util_raw
  fi
fi

if (( has_active_connections == 1 || gpu_util > GPU_UTIL_THRESHOLD )); then
  echo \"\$now_epoch\" > \"\$STATE_FILE\"
  exit 0
fi

last_active_epoch=\$(cat \"\$STATE_FILE\" 2>/dev/null || echo \"\$now_epoch\")
if [[ ! \"\$last_active_epoch\" =~ ^[0-9]+$ ]]; then
  last_active_epoch=\$now_epoch
fi

idle_seconds=\$((now_epoch - last_active_epoch))
if (( idle_seconds >= IDLE_MINUTES * 60 )); then
  logger -t helixserve-idle \"Idle for \${idle_seconds}s (gpu=\${gpu_util}%, port=\${CHECK_PORT}) -> shutting down\"
  /sbin/shutdown -h now \"HelixServe idle shutdown (\${IDLE_MINUTES}m)\"
fi
EOF

sudo chmod +x /usr/local/bin/helixserve_idle_shutdown.sh

cat <<EOF | sudo tee /etc/default/helixserve-idle >/dev/null
IDLE_MINUTES=${IDLE_MINUTES}
CHECK_PORT=${CHECK_PORT}
GPU_UTIL_THRESHOLD=${GPU_UTIL_THRESHOLD}
MIN_UPTIME_MINUTES=${MIN_UPTIME_MINUTES}
STATE_FILE=/var/lib/helixserve/last_active_epoch
DISABLE_FILE=/var/lib/helixserve/disable_idle_shutdown
EOF

cat <<'EOF' | sudo tee /etc/systemd/system/helixserve-idle-check.service >/dev/null
[Unit]
Description=HelixServe idle shutdown check
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/helixserve_idle_shutdown.sh
EOF

cat <<'EOF' | sudo tee /etc/systemd/system/helixserve-idle-check.timer >/dev/null
[Unit]
Description=Run HelixServe idle shutdown checks every 5 minutes

[Timer]
OnBootSec=10m
OnUnitActiveSec=5m
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now helixserve-idle-check.timer
sudo systemctl start helixserve-idle-check.service || true
sudo systemctl status helixserve-idle-check.timer --no-pager
"

echo "Configured idle shutdown on $INSTANCE_NAME (idle=${IDLE_MINUTES}m, port=${CHECK_PORT}, gpu-threshold=${GPU_UTIL_THRESHOLD}%)"
