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

parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--rm", type=str, default="metricx24") # "Unbabel/wmt22-comet-da"
parser.add_argument("--llm_gpu", type=str, default="cuda:0")
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--start", type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument("--dataset", type=str, default='/home/raychen/20240729/datasets/zh_en_train_llama3_gemma2.csv')
parser.add_argument("--language", type=str, choices=['en', 'de', 'ru'], default='en')
parser.add_argument('--type', type=str, choices=['paragraph', 'context'], default='paragraph')
parser.add_argument("--max_new_token", type=int, default=1024)
parser.add_argument("--recover", action='store_true', default = False)
parser.add_argument("--config", type=str, default="args/greedy_rm_1.5.jsonl" )
parser.add_argument("--out_file", type=str, default="args/run_outs")
args = parser.parse_args()

print(f"{args=}")

if args.recover:
    print("[INFO]: LOOKS LIKE YOU WANT TO RECOVER SOME RESULTS,")
    print("[INFO]: MAKE SURE ALL COMMANDLINE ARGS ARE EXACTLY THE SAME!!!")
    input("PRESS ENTER TO CONTINUE")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: Config doesn't exist!")
    exit(1)
    
out_path = Path(args.out_file + f"/0_zh-{args.language}_{args.type}.jsonl")
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

test_ds = pd.read_csv(args.dataset)
src = test_ds['zh'].replace('</s>', '')
ref = test_ds[args.language].replace('</s>', '')

truncated_ds = []
# for i in range(args.start, args.end_idx):
for i in range(len(test_ds)):
    truncated_ds.append([src[i],ref[i]])
print(f"{len(truncated_ds)=}")

print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
search = ARGS(llm_path=args.llm, rm_path=args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
print(f"[INFO]: Done")

def runprompt(src, ref, prompt: str, rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sample_temp=None, llm_dev:str="cuda:0") -> str:
    tokens, call = search.generate(src, ref, prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokenizer.decode(tokens[0], skip_special_tokens=True)
    del tokens
    # tokens_text_np = tokens_text.removeprefix(prompt)
    return tokens_text, raw_tokens, call

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(args.out_file + f"/{config_num}_zh-{args.language}_{args.type}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"/{config_num}_zh-{args.language}_{args.type}.jsonl"))
        samples = resfile.readlines()

        if samples[-1] != "":
            print("last line not empty??")
            exit(1)
        
        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) -1]:
            print(f"[INFO]: PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        if args.recover and (idx <= len(samples) -1):
            print(f"[INFO]: SKIPPING {idx}")
            continue
        
        language_map = {'en': 'English', 'de': 'German', 'ru': 'Russian'}
        language = language_map.get(args.language, '')
        src = ds_row[0]
        ref = ds_row[1]
        system_prompt = [{"role": "system", "content": f"You are a helpful translator and only output the result. Translate this from Chinese to {language}:\n "}]
        prompt = [{"role": "user", "content": src}]
        current_prompt = system_prompt + prompt
        print(f'{current_prompt=}')
        
        start = time.time()
        # src, ref
        res, tokens, call = runprompt(src, ref, current_prompt, float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sample_temp"], llm_dev=args.llm_gpu)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        data.append({"prompt": current_prompt, "result": res, "elapsed":elapsed,"call": call}) # , "method": args.out_file + f"_{config_num}"
        print(data)
        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"/{config_num}_zh-{args.language}_{args.type}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
