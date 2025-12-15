# Evaluate VLMs on Aksara

Tasks:
1. Image-to-text Transliteration
2. Text-to-Image Generation (still WIP)

Supported language:
1. Javanese
2. Balinese

Next: Sundanese, Lampung

## Setup Environment

OS: Linux
CUDA Ver: 12.x

1. Setup `.env` by modifying `.env.example`
2. Setup the conda env: `conda env create -f environment.yml`

## Running the image-to-text transliteration script

To run the evaluation script, check `script/run_internvl.sh`.

1. Modifiy the `MODEL` and `LANG` var.
2. Currently supported `LANG`: `"javenese"` and `"balinese"`.
3. run it with `bash script/run_internvl.sh`

## Compute the score

To run the compute score script, run `bash script/compute_score.sh`.

## **Disclaimer!!**

The script will also run a **vLLM** in the background, so you probably need to kill the process afterwards!!
you can see the PID with `nvidia-smi` and use `kill` to stop it.