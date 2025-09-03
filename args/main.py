from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from argsearch import ARGS
import time
import os
import pandas as pd
import torch
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--rm", type=str, default='rl-bandits-lab/helpsteer_rm')
parser.add_argument("--max_new_token", type=int, default=1024)
parser.add_argument("--recover", action='store_true', default=False)
parser.add_argument("--config", type=str, default="./baseline/args/greedy_rm_1.5.jsonl" )
parser.add_argument("--out_file", type=str, default="./baseline/args/run_outs")

parser.add_argument('-l',"--llm_gpu", type=str, default="cuda:0")
parser.add_argument('-r',"--rm_gpu", type=str, default="cuda:0")
parser.add_argument('-s',"--start", type=int, default=100)
parser.add_argument('-e','--end', type=int, default=694)
parser.add_argument('-d', '--dataset', type=str, default='/home/raychen/20241202/helpsteer/dataset/helpsteer3_general_test_valid_only.csv')
args = parser.parse_args()

print(f"{args=}")
df = pd.read_csv(args.dataset)
df['prompt'] = df['prompt'].apply(ast.literal_eval)
if args.start == 0 and (args.end == -1 or args.end > len(df)):
    iterator_obj = range(len(df))
elif args.end == -1 or args.end > len(df):
    iterator_obj = range(args.start,len(df))
else:
    iterator_obj = range(args.start,args.end)
print(f"[INFO]: Loaded dataset {args.dataset}, {iterator_obj}")
ds = df[iterator_obj.start:iterator_obj.stop]

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: Config doesn't exist!")
    exit(1)

out_path = Path(f"{args.out_file}/args_{args.dataset}_{iterator_obj.start}-{iterator_obj.stop}.jsonl")
if out_path.exists() and (not args.recover):
    print("ERROR: out_path already exists!")
    exit(1)

if not out_path.exists() and args.recover:
    print("ERROR: out_path DOESN'T exist!")
    exit(1)

with open(cfg_path) as f:
    run_configs = [json.loads(line) for line in f.readlines()]
    
# validate configs
for run_config in run_configs:
    if "rm_weight" not in run_config:
        print(f"Missing key 'rm_weight' in {run_config=}")
        exit(1)
    elif "topk" not in run_config:
        print(f"Missing key 'topk' in {run_config=}")
        exit(1)
    elif "mode" not in run_config:
        print(f"Missing key 'mode' in {run_config=}")
        exit(1)
    elif "sample_temp" not in run_config:
        print(f"Missing key 'sample_temp' in {run_config=}")
        exit(1)

print(f"[INFO]: Loaded {len(run_configs)} run configs.")
print(f"[DEBUG]: {run_configs=}")

print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
search = ARGS(llm_path=args.llm, rm_path=args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
print(f"[INFO]: Done")

def runprompt(prompt, rm_weight=0., topk=5, new_token=24, mode="greedy_large", sample_temp=None, llm_dev="cuda:0") -> str:
    tokens, call = search.generate(prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokenizer.decode(tokens[0], skip_special_tokens=True)
    del tokens
    return tokens_text, raw_tokens, call

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(f"{args.out_file}/args_{args.dataset}_{iterator_obj.start}-{iterator_obj.stop}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(f"{args.out_file}/args_{args.dataset}_{iterator_obj.start}-{iterator_obj.stop}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("last line not empty??")
            exit(1)
        
        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != ds[len(samples) -1]:
            print(f"[INFO]: PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    for idx, prompt in enumerate(tqdm(ds['prompt'])):
        if args.recover and (idx <= len(samples) -1):
            print(f"[INFO]: SKIPPING {idx}")
            continue

        start = time.time()
        # src, ref
        res, tokens, call = runprompt(prompt, float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sample_temp"], llm_dev=args.llm_gpu)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        print({"prompt": prompt, "result": res, "elapsed":elapsed,"call": call})
        data.append({"prompt": prompt, "result": res, "elapsed":elapsed,"call": call})
        with open(Path(f"{args.out_file}/args_{iterator_obj.start}-{iterator_obj.stop}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
