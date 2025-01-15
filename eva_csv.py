import csv
import json
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-da","--dataset", type=str, default='/home/raychen/20240729/baseline_acl2025/simpo_out')
parser.add_argument("-j","--json", type=str, default='/home/raychen/20240729/baseline_acl2025/simpo_out/json')
parser.add_argument("-f","--fig", type=str, default='/home/raychen/20240729/baseline_acl2025/simpo_out/fig')
parser.add_argument("-l","--lang", type=str, choices=['en','de','ru'], default='en')
parser.add_argument("-m","--model", type=str, default='SimPO')
parser.add_argument("-d","--device", type=str, default="cuda:0")
parser.add_argument("-r","--reference", type=bool, default=False)
args = parser.parse_args()


def get_entry(row, model, lang, reference=False):
    entry = {"source": row['zh'].replace('</s>', '')}
    if model=='ref':
        entry["hypothesis"] = row[lang].replace('</s>', '')
    else:
        entry["hypothesis"] = row[model].replace('</s>', '')
    if reference:
        entry["reference"] = row[lang].replace('</s>', '')
    else:
        entry["reference"] = ""
    return entry

def write_jsonl(json_path, model, lang, reader, reference=False):
    r = 'ref-based' if reference else 'ref-free'
    with open(f'{json_path}/input_zh-{lang}_{model}_{r}.jsonl', 'w', encoding='utf-8') as output_file:
        for row in reader:
            entry = get_entry(row, model, lang, reference)
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
def run_command(json_path, model, lang, reference=False, context=False, device='cuda:0'):
    r = 'ref-based' if reference else 'ref-free'
    # c = '_context' if context else ''
    command = [
        "python", "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", "google/metricx-24-hybrid-xl-v2p6",
        "--max_input_length", "1536",
        "--batch_size", "1",
        "--input_file", f"{json_path}/input_zh-{lang}_{model}_{r}.jsonl",
        "--output_file", f"{json_path}/output_zh-{lang}_{model}_{r}.jsonl",
        "--device", device,
    ]
    if reference==False:
        command.append("--qe")
    subprocess.run(command)

def get_predict(model, lang, reference):
    r = 'ref-based' if reference else 'ref-free'
    scores = []
    with open(f"{args.json}/output_zh-{lang}_{model}_{r}.jsonl", 'r', encoding='utf-8') as new:
        for line in new:
            entry = json.loads(line)
            score = entry.get('prediction', None)
            scores.append(score)
    return scores

def mean_and_plot(model, lang, fig_path, reference=False):
    r = 'ref-based' if reference else 'ref-free'
    scores = get_predict(model, lang, reference)
    print(f'{np.mean(scores):.2f}±{np.std(scores):.2f}')
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, alpha=0.9)
    plt.title(f'Scores Distribution of {model}, zh-{lang}, {r}, paragraph')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{fig_path}/zh-{lang}_{model}_{r}.png')

with open(f'{args.dataset}/zh-{args.lang}_paragraph.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    write_jsonl(args.json, args.model, args.lang, reader, args.reference)
    
run_command(args.json, args.model, args.lang, args.reference, False, args.device)
mean_and_plot(args.model, args.lang, args.fig, args.reference)