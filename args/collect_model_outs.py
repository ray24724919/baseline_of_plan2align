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
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())


# os.environ["TORCH_USE_CUDA_DSA"] = '0,1,2,3'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='translation/en-zh') # "translation/en-zh"
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--run_percent", type=float, default=.16)
# parser.add_argument("--rm", type=str, default="argsearch/llama-7b-rm-float32")
# parser.add_argument("--llm", type=str, default="argsearch/llama-7b-sft-float32")
parser.add_argument("--rm", type=str, default="Unbabel/wmt22-comet-da") # "Unbabel/XCOMET-XL"
parser.add_argument("--llm", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--max_new_token", type=int, default=1024)

parser.add_argument("--llm_gpu", type=str, default="cuda:1")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")

parser.add_argument("--start_idx", type=int, default=60)
parser.add_argument("--recover", action='store_true', default = False)

parser.add_argument("--config", type=str, default="greedy_rm_1.5.jsonl" )

parser.add_argument("--out_file", type=str, default="run_outs/0106_2")

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
    
out_path = Path(args.out_file + f"_0.jsonl")
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
    
print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")

if args.dataset == "Dahoas/full-hh-rlhf":
    # FOR HHRLHF
    test_ds = load_dataset(args.dataset, split=args.split)
    test_ds = test_ds["prompt"]
elif args.dataset == "stanfordnlp/SHP":
    # FOR SHP
    test_ds = load_dataset(args.dataset, split=args.split)
    unique_prompts = []
    seen_posts = set()
    for post_id, histr in zip(test_ds["post_id"], test_ds['history']):
        if post_id in seen_posts: continue
        model_prompt = " Human: " + histr + " Assistant: "
        unique_prompts.append(model_prompt)
        seen_posts.add(post_id)
    test_ds = unique_prompts
elif args.dataset == "translation/en-zh":
    # test_ds = pd.read_csv('./dataset/small_dataset_segment.csv')
    try:
        test_ds = pd.read_csv('/home/raychen/20241202/dataset/wmt24_zh_en_split_train.csv')
    except:
        test_ds = pd.read_csv('/home/raychen/20240729_llama/datasets/wmt24_zh_en_split_train.csv')
    src = test_ds['en'].replace('</s>', '')
    ref = test_ds['zh'].replace('</s>', '')
elif args.dataset == "translation/zh-en":
    test_ds = pd.read_csv('/home/raychen/20240729_llama/args_new/preference_dataset.csv')
    src = test_ds['ref'].replace('</s>', '')
    ref = test_ds['src'].replace('</s>', '')

end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

# truncated_ds = test_ds[0:end_idx]
# print(f"{len(truncated_ds)=}")
start_idx = args.start_idx
truncated_src = src[0:end_idx].replace('</s>', '')
truncated_ref = ref[0:end_idx].replace('</s>', '')
truncated_ds = []
for i in range(start_idx,end_idx):
    truncated_ds.append([truncated_src[i],truncated_ref[i]])
print(f"{len(truncated_ds)=}")

print(f"[INFO]: Loading models ({args.llm=}, {args.rm=})")
search = ARGS(llm_path=args.llm, rm_path=args.rm, llm_dev=args.llm_gpu, rm_dev=args.rm_gpu)
print(f"[INFO]: Done")

def runprompt(src, ref, prompt: str, rm_weight=0., topk=5, new_token=24, mode="p_sigmoid_mixing", sample_temp=None, llm_dev:str="cuda:0") -> str:
    tokens, call = search.generate(src, ref, prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)

    # too long seqlen
    if tokens == None: return None, None
    
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    #tokens_text = search.tokens_to_text(tokens)[0]
    tokens_text = search.tokenizer.decode(tokens[0], skip_special_tokens=True)
    del tokens
    # tokens_text_np = tokens_text.removeprefix(prompt)
    return tokens_text, raw_tokens, call

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
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

        # print(f"{ds_row=}")
        current_prompt = ds_row #["prompt"]
        if args.dataset == "translation/en-zh" or "translation/zh-en":
            src = str(ds_row[0]).replace('</s>', '')
            ref = str(ds_row[1]).replace('</s>', '')
            if args.dataset == "translation/en-zh":
                system_prompt = [{"role": "system", "content": "你是一個非常有經驗且成功的譯者，你的任務是將給定的 text 翻譯成繁體中文，並且只輸出翻譯結果。"}]
                # system_prompt = "你是一個非常有經驗且成功的譯者，你的任務是將以下給定的文章翻譯成繁體中文，並且只輸出翻譯結果:\n"
                # system_prompt_back = "\n翻譯結果:"
            elif args.dataset == "translation/zh-en":
                system_prompt = [{"role": "system", "content": "You are a helpful translator, your task is to translate the given text from Chinese to English and only output the result."}]
                # system_prompt = "You are a helpful translator, your task is to translate the given text from Chinese to English and only output the result:\n"
                # system_prompt_back = "\ntranslation result:"
            prompt = [{"role": "user", "content": src}]
            # prompt = src
            # current_prompt = system_prompt + prompt  + system_prompt_back
            current_prompt = system_prompt + prompt
        print(f'{current_prompt=}')
        
        start = time.time()
        # src, ref
        res, tokens, call = runprompt(src, ref, current_prompt, float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sample_temp"], llm_dev=args.llm_gpu)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        data.append({"prompt": current_prompt, "result": res, "elapsed":elapsed,"call": call, "method": args.out_file + f"_{config_num}"})
        print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"_{config_num}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
