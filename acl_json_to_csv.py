import json
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lang", type=str, choices=['en','de','ru'], default='en')
parser.add_argument("-s","--set_name", type=str, choices=['train','validation'], default="validation")
parser.add_argument('-c', '--context', type=str, choices=['paragraph','context'], default='paragraph')
parser.add_argument("-m","--model", type=str, default='rain')
parser.add_argument("-d","--device", type=str, default="cuda:0")
args = parser.parse_args()
print(f"{args=}")

path = os.getcwd() + f'/{args.model}/results'
if not os.path.exists(path):
    os.makedirs(path)

output_path = path + f'/zh-{args.lang}_{args.context}.csv'
dataset_path = f'/home/raychen/20240729/acl_datasets/{args.set_name}/{args.context}/acl2025_{args.set_name}_zh-{args.lang}_{args.context}.csv'
df = pd.read_csv(dataset_path)
print(f'{len(df)=}')
language_map = {'en': 'English', 'de': 'German', 'ru': 'Russian'}
language = language_map.get(args.lang, '')
src = df['zh']
ref = df[args.lang]

def get_rain(df, language):
    input_path = os.getcwd() + f'/{args.model}/run_outs/res_0_zh-{args.lang}_{args.context}.json'
    texts = []
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        front = f'You are a helpful translator and only output the result. Translate this from Chinese to {language}:\n You are a helpful translator and only output the result. Translate this from Chinese to {language}:\n '
        for i in range(len(df)):
            tmp = data[i]["raina"].replace(front,'')
            tmp = tmp.replace(src[i],'')
            tmp = tmp.replace('Translation result:','')
            texts.append(tmp)
    return texts

def get_args(df, language):
    input_path = os.getcwd() + f'/{args.model}/run_outs/0_zh-{args.lang}_{args.context}.jsonl'
    texts = []
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        front = f'system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a helpful translator and only output the result. Translate this from Chinese to {language}:user\n\n'
        for i in range(len(df)):
            tmp = data[i]["result"].replace(front,'')
            tmp = tmp.replace(src[i],'')
            tmp = tmp.replace('Translation result:','')
            tmp = tmp.replace('assistant\n\n','')
            texts.append(tmp)
    return texts

if args.model=='rain':
    texts = get_rain(df, language)
elif args.model=='args':
    texts = get_args(df, language)

df[args.model] = texts
df.to_csv(output_path, index=False)

dd = pd.read_csv(output_path)
print(len(dd))