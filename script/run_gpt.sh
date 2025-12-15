#!/bin/bash

set -a
source ".env"
set +a

MODEL="gpt-5-nano"
LANG="javanese" # or LANG="balinese"

DATAPATH="izzako/$LANG-pixelgpt-test"
SHORTMODEL="${MODEL##*/}"

PROMPT="Transliterate the text in the image into Latin (romanized) script. 
Return ONLY the transliterated result. 
If the image does not contain translatable script text, return an empty string."

LOG_DIR="logs/${SHORTMODEL}"
OUT_DIR="outputs/${SHORTMODEL}/${LANG}"

mkdir -p "$LOG_DIR"

echo "Starting the Simulation..."

python -u src/benchmark_image_transliterate.py \
  --dataset_path "${DATAPATH}" \
  --split "test" \
  --model_name "${MODEL}" \
  --base_url "https://api.openai.com/v1" \
  --api-key "$OPENAI_API_KEY" \
  --source_language "${LANG}" \
  --prompt "${PROMPT}" \
  --output_folder "$OUT_DIR" \
  --fewshot_file "example/fewshot_$LANG.json" \
  --temperature 0.4 \
  --batch_size 5 \
  --max_tokens 2048 \
  &> "$LOG_DIR/benchmark_$LANG.log"