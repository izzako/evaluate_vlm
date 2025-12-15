import jiwer
import sacrebleu
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='VLM Compute Score Tool')
    parser.add_argument('--results_folder', type=str, required=True, 
                       help='Folder to load VLM output results')
    parser.add_argument('--score_folder', type=str, required=True, 
                       help='Folder to save computed score results') 

    args = parser.parse_args()

    all_metrics = []
    outdir = args.results_folder
    scoredir = args.score_folder

    print(f"VLM Output Results Folder: {outdir}")
    print(f"Score Folder: {scoredir}")

    for model in tqdm(os.listdir(outdir)):
        for lang in tqdm(os.listdir(os.path.join(outdir,model)),leave=False):
            for filename in os.listdir(os.path.join(outdir,model,lang)):
                jsonfile = os.path.join(outdir,model,lang,filename)
                dt = datetime.strptime(filename.lstrip('results_test_').rstrip('.jsonl'), "%Y%m%d_%H%M%S")
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                data = []
                with open(jsonfile, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line))
                all_references = [i['metadata']['label_text'] for i in data]
                all_predictions = [i['response'] for i in data]
                wer = jiwer.wer(all_references, all_predictions)
                bleu = sacrebleu.corpus_bleu(all_predictions, [[ref] for ref in all_references])
                chrfpp = sacrebleu.corpus_chrf(all_predictions, [[ref] for ref in all_references], word_order=2)

                metrics = {
                    "experiment_datetime":formatted,
                    "model":model,
                    "language":lang.title(),
                    "wer": wer * 100,
                    "bleu": bleu.score,
                    "chrF++": chrfpp.score}
                
                all_metrics.append(metrics)
    os.makedirs(scoredir,exist_ok=True)
    outpath = f'{scoredir}/image_to_text_transliteration.json'
    json.dump(all_metrics,open(outpath,'w'),indent=4,ensure_ascii=False)

    print(f"Scores saved in {outpath}")                  

if __name__ == "__main__":
    main()