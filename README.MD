# Evaluate VLMs on Aksara

## Setup Environment

CUDA Ver: 12.x

1. Setup `.env` by modifying `.env.example`
2. Setup the conda env: `conda env create -f environment.yml`

## Running the program

To run the evaluation script, check `script/run_internvl.sh`.

1. Modifiy the `MODEL` and `LANG` var.
2. Currently supported `LANG`: `"javenese"` and `"balinese"`.
3. run it with `bash script/run_internvl.sh`

## Disclaimer

The script will also run a **vLLM** in the background, so u prolly need to kill the process afterwards!!
u can see the PID with `nvidia-smi` and use `kill` to stop it.