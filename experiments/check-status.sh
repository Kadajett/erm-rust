#!/bin/bash
# ERM Experiment Status Checker
# Outputs STATUS.json with per-experiment metrics

STATUS_FILE="/home/node/.openclaw/workspace/erm-rust/experiments/STATUS.json"

echo "{"
echo '  "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",'
echo '  "experiments": ['

FIRST=true
for EXP in exp-a exp-b exp-c exp-d exp-e exp-f; do
  POD=$(kubectl get pods -n pcn-train -l exp=${EXP} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  STATUS=$(kubectl get pods -n pcn-train -l exp=${EXP} -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
  
  if [ -z "${STATUS}" ]; then
    STATUS="Unknown"
  fi
  
  STEP="0"
  LOSS="null"
  
  if [ "${STATUS}" = "Running" ] || [ "${STATUS}" = "Succeeded" ]; then
    LAST_LINE=$(kubectl logs "${POD}" -n pcn-train --tail=1 2>/dev/null || echo "")
    if echo "${LAST_LINE}" | grep -q 'colony step'; then
      STEP=$(echo "${LAST_LINE}" | grep -oP 'step\s+\K\d+')
      LOSS=$(echo "${LAST_LINE}" | grep -oP 'loss=\K[0-9.]+')
    fi
  fi
  
  # Estimate steps/sec from last few log lines
  RATE="null"
  ETA="null"
  if [ "${STATUS}" = "Running" ] && [ "${STEP}" -gt 50 ] 2>/dev/null; then
    # Get timestamp of step 0 and current from logs
    LINES=$(kubectl logs "${POD}" -n pcn-train 2>/dev/null | grep 'colony step' | tail -5)
    if [ -n "${LINES}" ]; then
      REMAINING=$((10000 - STEP))
      # rough: ~50 steps per log entry, get time between recent entries
      START_TIME=$(kubectl logs "${POD}" -n pcn-train 2>/dev/null | grep 'Start:' | grep -oP '\d{2}:\d{2}:\d{2}' | head -1)
      if [ -n "${START_TIME}" ]; then
        NOW=$(date -u +%H:%M:%S)
        START_SEC=$(date -u -d "${START_TIME}" +%s 2>/dev/null || echo 0)
        NOW_SEC=$(date -u +%s)
        if [ "${START_SEC}" -gt 0 ] 2>/dev/null; then
          ELAPSED=$((NOW_SEC - START_SEC))
          if [ "${ELAPSED}" -gt 0 ] && [ "${STEP}" -gt 0 ]; then
            RATE=$(echo "scale=2; ${STEP} / ${ELAPSED}" | bc)
            ETA_SEC=$(echo "scale=0; ${REMAINING} / (${STEP} / ${ELAPSED})" | bc 2>/dev/null)
            ETA_MIN=$(echo "scale=1; ${ETA_SEC} / 60" | bc 2>/dev/null)
            ETA="${ETA_MIN}min"
          fi
        fi
      fi
    fi
  fi
  
  if [ "${FIRST}" = "true" ]; then
    FIRST=false
  else
    echo ","
  fi
  
  printf '    {"exp": "%s", "pod": "%s", "status": "%s", "step": %s, "loss": %s, "rate_steps_sec": %s, "eta": "%s"}' \
    "${EXP}" "${POD}" "${STATUS}" "${STEP:-0}" "${LOSS:-null}" "${RATE:-null}" "${ETA:-unknown}"
done

echo ""
echo "  ]"
echo "}"
