import csv
import json
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lang", type=str, choices=['en','de','ru'], default='en')
parser.add_argument("-s","--set_name", type=str, choices=['train','validation'], default="train")
parser.add_argument('-c', '--context', type=str, choices=['paragraph','context'], default='paragraph')
parser.add_argument("-m","--model", type=str, default='gpt4o')
parser.add_argument("-d","--device", type=str, default="cuda:0")
args = parser.parse_args()
print(f"{args=}")

# to /baseline_acl_2025 first
path = os.getcwd() + f'/{args.model}/results'
dataset_path = f'/home/raychen/20240729/acl_datasets'
if args.model=='gpt4o':
    res_path = f'{dataset_path}/{args.set_name}/{args.context}/acl2025_{args.set_name}_zh-{args.lang}_{args.context}.csv'
else:
    res_path = f'{path}/zh-{args.lang}_{args.context}.csv' # for rain, args

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

def write_jsonl(path, model, lang, reader, reference=False, context='paragraph'):
    r = 'ref-based' if reference else 'ref-free'
    with open(f'{path}/input_zh-{lang}_{model}_{context}_{r}.jsonl', 'w', encoding='utf-8') as output_file:
        for row in reader:
            entry = get_entry(row, model, lang, reference)
            output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
def run_command(path, model, lang, reference=False, context='paragraph', device='cuda:0'):
    r = 'ref-based' if reference else 'ref-free'
    devices_map = {'cuda:0':0,'cuda:1':1,'cuda:2':2,'cuda:3':3}
    command = [
        "python", "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", "google/metricx-24-hybrid-xl-v2p6",
        "--max_input_length", "1536",
        "--batch_size", "1",
        "--input_file", f"{path}/input_zh-{lang}_{model}_{context}_{r}.jsonl",
        "--output_file", f"{path}/output_zh-{lang}_{model}_{context}_{r}.jsonl",
        "--device", f'{devices_map.get(device, 0)}',
    ]
    if reference==False:
        command.append("--qe")
    subprocess.run(command)

def get_predict(model, lang, reference=False, context='paragraph'):
    r = 'ref-based' if reference else 'ref-free'
    scores = []
    with open(f"{path}/output_zh-{lang}_{model}_{context}_{r}.jsonl", 'r', encoding='utf-8') as new:
        for line in new:
            entry = json.loads(line)
            score = entry.get('prediction', None)
            scores.append(score)
    return scores

def mean_and_plot(model, lang, path, reference=False, context='paragraph'):
    r = 'ref-based' if reference else 'ref-free'
    scores = get_predict(model, lang, reference)
    print(f'Scores of {model}, zh-{lang}, {context}, {r}: {np.mean(scores):.2f}±{np.std(scores):.2f}')
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, alpha=0.9)
    plt.title(f'Scores Distribution of {model}, zh-{lang}, {context}, {r}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{path}/zh-{lang}_{model}_{r}.png')
    return scores

with open(res_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    write_jsonl(path=path, model=args.model, lang=args.lang, reader=reader, reference=True)

with open(res_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    write_jsonl(path=path, model=args.model, lang=args.lang, reader=reader, reference=False)

run_command(path=path, model=args.model, lang=args.lang, reference=True, context='paragraph', device=args.device)
run_command(path=path, model=args.model, lang=args.lang, reference=False, context='paragraph', device=args.device)

scores = mean_and_plot(model=args.model, lang=args.lang, path=path, reference=True, context='paragraph')
scores_qe = mean_and_plot(model=args.model, lang=args.lang, path=path, reference=False, context='paragraph')

df = pd.read_csv(res_path)
df = df[['BOOK_ID','CHAPTER_ID','PARAGRAPH_ID','zh',args.lang,args.model]]
    
df['metricX'] = scores
df['metricX-qe'] = scores_qe
if args.set_name=='validation':
    output_path = f'{dataset_path}/score/{args.set_name}/{args.model}'
else:
    output_path = f'{dataset_path}/score/{args.set_name}/{args.context}'
if not os.path.exists(output_path):
    os.makedirs(output_path)
df.to_csv(output_path + f'/score_acl2025_{args.set_name}_zh-{args.lang}_{args.context}_{args.model}.csv', index=False)