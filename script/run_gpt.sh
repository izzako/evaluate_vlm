#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=12:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N eval_aksara_gpt

# Combine output and error files into a single file
#$ -j y

# Request 2 core
#$ -pe omp 2

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="

module load cuda/12.2 gcc/12.2.0 miniconda/23.11.0

conda activate "eval_aksara"

echo "Using Python: $(which python)"

set -a
source ".env"
set +a

MODEL="gpt-5-nano"
SHORTMODEL="${MODEL##*/}"

PROMPT="Transliterate the text in the image into Latin (romanized) script. 
Return ONLY the transliterated result. 
If the image does not contain translatable script text, return an empty string."

LOG_DIR="logs/${SHORTMODEL}"

mkdir -p "$LOG_DIR"

echo "Starting the Simulation..."

LANGS=('sundanese' 'lampung') #'javanese' 'balinese'

for LANG in "${LANGS[@]}"; do
  echo "Running on language: ${LANG}"
  OUT_DIR="outputs/${SHORTMODEL}/${LANG}"
  DATAPATH="izzako/$LANG-pixelgpt-test"
  python -u src/benchmark_image_transliterate.py \
    --dataset_path "${DATAPATH}" \
    --split "test" \
    --model_name "${MODEL}" \
    --base_url "https://api.openai.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --source_language "${LANG}" \
    --prompt "${PROMPT}" \
    --output_folder "$OUT_DIR" \
    --fewshot_file "example/fewshot_$LANG.json" \
    --temperature 1 \
    --batch_size 5 \
    --max_tokens 2048 \
    &> "$LOG_DIR/benchmark_$LANG.log"
done