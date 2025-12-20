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
#$ -N eval_aksara_qwen

# Combine output and error files into a single file
#$ -j y

# Request 2 core
#$ -pe omp 2

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0


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

MODEL="Qwen/Qwen3-VL-8B-Instruct"

PROMPT="Transliterate the text in the image into Latin (romanized) script. 
Return ONLY the transliterated result. 
If the image does not contain translatable script text, return an empty string."

SHORTMODEL="${MODEL##*/}"
LOG_DIR="logs/${SHORTMODEL}"
mkdir -p "$LOG_DIR"

nohup vllm serve "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt.video 0 \
    --max-model-len 128000 \
    --limit-mm-per-prompt.image 10 \
    > "$LOG_DIR/vllm.log" 2>&1 &

echo "Starting vLLM for $MODEL (logs in $LOG_DIR)..."

# Wait Loop
until curl -s http://localhost:8000/v1/models | grep -q "id"; do
    echo "Waiting for model to load... Retrying in 1 minute"
    sleep 60
done

echo "vLLM is ready! Starting the Simulation..."

LANGS=('sundanese' 'lampung' 'balinese') #('javanese' 'balinese')

for LANG in "${LANGS[@]}"; do
  echo "Running on language: ${LANG}"
  OUT_DIR="outputs/${SHORTMODEL}/${LANG}"
  DATAPATH="izzako/$LANG-pixelgpt"
  python -u src/qwen_benchmark_it.py \
    --dataset_path "${DATAPATH}" \
    --split "test" \
    --model_name "${MODEL}" \
    --source_language "${LANG}" \
    --prompt "${PROMPT}" \
    --output_folder "$OUT_DIR" \
    --fewshot_file "example/fewshot_$LANG.json" \
    --temperature 0.4\
    --batch_size 5 \
    --max_tokens 2048 \
    &> "$LOG_DIR/benchmark_$LANG.log"
done