#!/bin/bash

set -a
source ".env"
set +a

MODEL="OpenGVLab/InternVL3_5-8B"
LANG="balinese" # or LANG="javanese"
DATAPATH="izzako/$LANG-pixelgpt-test"
SHORTMODEL="${MODEL##*/}"

PROMPT="Transliterate the text in the image into Latin (romanized) script. 
Return ONLY the transliterated result. 
If the image does not contain translatable script text, return an empty string."

LOG_DIR="logs/${SHORTMODEL}"
OUT_DIR="outputs/${SHORTMODEL}/${LANG}"

mkdir -p "$LOG_DIR"

nohup vllm serve "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    > "$LOG_DIR/vllm.log" 2>&1 &

echo "Starting vLLM for $MODEL (logs in $LOG_DIR)..."

# Wait Loop
until curl -s http://localhost:8000/v1/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 1 minute"
    sleep 60
done

echo "vLLM is ready! Starting the Simulation..."

python -u src/benchmark_image_transliterate.py \
  --dataset_path "${DATAPATH}" \
  --split "test" \
  --model_name "${MODEL}" \
  --source_language "${LANG}" \
  --prompt "${PROMPT}" \
  --output_folder "$OUT_DIR" \
  --fewshot_file "example/fewshot_$LANG.json" \
  --batch_size 5 \
  --max_tokens 2048 \
  &> "$LOG_DIR/benchmark_$LANG.log"