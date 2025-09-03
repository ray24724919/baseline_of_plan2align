import argparse
import os
import copy
import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--modelname', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('-g','--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('-s',"--start", type=int, default=0)
parser.add_argument('-e','--end', type=int, default=100)
parser.add_argument('-d', '--dataset', type=str, default='/home/raychen/20241202/helpsteer/dataset/helpsteer3_general_test_valid_only.csv')
parser.add_argument('--outdir', type=str, default='./baseline/rain/run_outs')
args = parser.parse_args()

print(f"{args=}")

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
maxlen = 1024
maxT = 20
minT = 10
Vt = 0.8

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

from nodes import node

def save_dict(dict_input, filename):
    print('save_dict')
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            dict_existing = json.load(file)
        dict_merged = {**dict_existing, **dict_input}
    else:
        dict_merged = dict_input

    with open(filename, 'w') as file:
        json.dump(dict_merged, file, ensure_ascii=False)

def find_all_indices(text, substring):
    indices = []
    start_index = 0
    while True:
        index = text.find(substring, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1
    return indices

with open('./baseline/rain/f2.txt') as f:
    fsred = f.read()

with open('./baseline/rain/r1.txt') as f:
    redA = f.read()

with open('./baseline/rain/r2.txt') as f:
    redB = f.read()



outdir = args.outdir
df = pd.read_csv(args.dataset)
# df['prompt'] = df['prompt'].apply(ast.literal_eval)

if args.start == 0 and (args.end == -1 or args.end > len(df)):
    iterator_obj = range(len(df))
elif args.end == -1 or args.end > len(df):
    iterator_obj = range(args.start,len(df))
else:
    iterator_obj = range(args.start,args.end)
print(f"[INFO]: Loaded dataset {args.dataset}, {iterator_obj}")
# ds = df[iterator_obj.start:iterator_obj.stop]

try:
    with open(f'{outdir}/rain_{iterator_obj.start}-{iterator_obj.stop}.json', encoding='utf-8') as f:
        res = json.loads(f.read())
    qs = []
    for i in res:
        qs.append(i['question'])
except:
    res = []
    qs = []

def build_dataset_rank(tokenizer, iterator_obj):
    ds = load_dataset('csv', data_files=args.dataset)['train']
    ds = ds.select(range(iterator_obj.start,iterator_obj.stop))
    def modify_prompt(example):
        text = ast.literal_eval(example['prompt'])
        # messages = [{"role": "user",   "content": text}]
        messages = text
        encoding = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return {
            "input_ids": encoding,
            "query": text,
            "queryf": text
        }

    ds = ds.map(
        modify_prompt,
        remove_columns=['prompt','chosen','rejected'],
        load_from_cache_file=False
    )

    ds.set_format(type="torch")
    return ds

if not os.path.exists(outdir):
    os.makedirs(outdir)
if not os.path.exists('{}_dicv'.format(outdir)):
    os.makedirs('{}_dicv'.format(outdir))

tokenizer = AutoTokenizer.from_pretrained(args.modelname, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    args.modelname,
    # load_in_4bit=True,
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    device_map="cuda:0",
)
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
testdataset = build_dataset_rank(tokenizer, iterator_obj)
# print(testdataset[0])
# testloader = DataLoader(testdataset, batch_size=1, shuffle=False)

# @torch.no_grad()
# def genc(s, model, tokenizer):
#     '''
#     vanilla autoregression
#     '''
#     input_ids = tokenizer(s, return_tensors="pt").input_ids
#     terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#     outs = model.generate(inputs=input_ids.cuda(), max_length=maxlen, use_cache=False, eos_token_id=terminators)
#     outstr = tokenizer.decode(outs[0], skip_special_tokens=True)
#     return outstr

model.eval()


@torch.no_grad()
def getv(getoken, model, tokenizer, dic, dicp, maxlen):
    '''
    score through self-evaluation
    '''
    text, simgstate = simg(dicp, getoken, model, tokenizer, maxlen)
    inds = find_all_indices(text, 'Human:')
    if len(inds) > 5:
        text = text[inds[4] : inds[5]]
    # 如果只有 5~6 之間（len == 5）的出現，就用第 4 號之後到結尾
    elif len(inds) > 4:
        text = text[inds[4] :]
    if text not in dic:
        textA_msg = [
            {"role":"system","content":fsred},
            {"role":"assistant","content":text},
            {"role":"user","content":redA}
        ]
        input_ids = tokenizer.apply_chat_template(
            textA_msg, return_tensors="pt"
        )
        textB_msg = [
            {"role":"system","content":fsred},
            {"role":"assistant","content":text},
            {"role":"user","content":redB}
        ]
        
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item()  # prob of 'A'
        p_B = prob[29933].item()  # prob of 'B'
        if p_A > p_B:
            A = 1
        else:
            A = 0
        input_ids = tokenizer.apply_chat_template(
            textB_msg, return_tensors="pt"
        )
        outs = model(input_ids.cuda())
        logits = outs.logits
        last_token_logits = logits[0, -1, :]
        prob = F.softmax(last_token_logits.float(), dim=0)
        p_A = prob[29909].item()
        p_B = prob[29933].item()
        if p_B > p_A:
            B = 1
        else:
            B = 0
        v = (A + B) / 2
        v = (v - 0.5) * 2
        dic[text] = v
    else:
        v = dic[text]
    return v, simgstate, len(simgstate) - len(getoken)

@torch.no_grad()
def simg(dicp, orstate, model, tokenizer, maxlen):
    '''
    simulation generation for more accurate self-evaluation
    '''
    state = copy.deepcopy(orstate)
    past_key_values = None
    while 1:
        if len(state) > maxlen:
            break
        if terminators[0] or terminators[1] in state:
            break
        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        if tmpstr[-1] == ',' or tmpstr[-1] == '.' or tmpstr[-1] == '?' or tmpstr[-1] == ':' or tmpstr[
            -1] == ';' or tmpstr[-1] == '\n':
            break
        inds = find_all_indices(tmpstr, 'Human:')
        if len(inds) > 1 + 4:
            break
        probs, past_key_values = getp(state, model, dicp, topk=-1, return_past_key_values=True,
                                      past_key_values=past_key_values)
        token = int(torch.multinomial(probs, num_samples=1))
        state.append(token)
        if token == terminators[0] or token == terminators[1]:
            break
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    return tmpstr, state

def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.no_grad()
def getp(state, model, dicp, topk=-1, topp=1.0, temperature=1.0, repetition_penalty=0.0, return_last_logits=False,
         return_past_key_values=False, past_key_values=None,):
    '''
    query LLM
    '''
    if tuple(state) not in dicp:
        if past_key_values != None:
            input_ids = torch.tensor([[state[-1]]])
            outs = model(input_ids.cuda(), past_key_values=past_key_values)
            increment()
        else:
            input_ids = torch.tensor([state])
            outs = model(input_ids.cuda())
            increment()
        logits = outs.logits
        past_key_values = outs.past_key_values
        last_logits = logits[:, -1, :].float().cpu()
        dicp[tuple(state)] = last_logits
    else:
        last_logits = dicp[tuple(state)]
        past_key_values = None

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, topp, topk
    )
    last_token_logits = logits_processor(None, last_logits)[0]
    probs = torch.softmax(last_token_logits, dim=-1)
    if return_last_logits and return_past_key_values:
        return probs, last_logits, past_key_values
    if return_last_logits:
        return probs, last_logits
    if return_past_key_values:
        return probs, past_key_values
    return probs

@torch.no_grad()
def group_getp(state, model, dicp, topk=10, maxnew=10):
    '''
    group query LLM
    '''
    outs = []
    outsset = []
    etmpp = []
    if maxnew == 1:
        p, last_logits = getp(state, model, dicp, topk=topk, return_last_logits=True)
        acp = p.cpu().detach().squeeze(0).numpy()
        legal = np.where(acp > 0)[0]
        acp = acp[legal]
        acp = zip(legal, acp)
        for ac, p in acp:
            outs.append(([ac], p))
        return outs, last_logits

    greedytmpstate = copy.deepcopy(state)
    greedytmplog = torch.tensor(0.0)
    greedytmptokens = []
    greedy_past_key_values = None
    for i in range(maxnew):
        greedyprobs,greedy_past_key_values = getp(greedytmpstate, model, dicp, topk=15,return_past_key_values=True,past_key_values=greedy_past_key_values)
        greedytoken = int(torch.argmax(greedyprobs))
        greedylogp = torch.log(greedyprobs[greedytoken])
        greedytmplog += greedylogp
        greedytmptokens.append(greedytoken)
        greedytmpstate.append(greedytoken)
    outsset.append(greedytmptokens)

    for _ in range(topk - 1):
        tmpstate = copy.deepcopy(state)
        tmplog = torch.tensor(0.0)
        tmptokens = []
        past_key_values = None
        for i in range(maxnew):
            probs,past_key_values = getp(tmpstate, model, dicp, topk=15,return_past_key_values=True,past_key_values=past_key_values)
            token = int(torch.multinomial(probs, num_samples=1))
            logp = torch.log(probs[token])
            tmplog += logp
            tmptokens.append(token)
            tmpstate.append(token)
        if tmptokens not in outsset:
            outsset.append(tmptokens)
            tmpp = torch.exp(tmplog)
            outs.append((tmptokens, tmpp.item()))
            etmpp.append(tmpp.item())
        if len(outs) >= topk - 1:
            break

    greedytmpp = torch.exp(greedytmplog)
    if len(etmpp)>0:
        etmpp=np.array(etmpp)
        greedytmpp = min(greedytmpp.item(), etmpp.sum())
        greedytmpp = max(greedytmpp, etmpp.max()+etmpp.min())
    else:
        greedytmpp=greedytmpp.item()
    outs = [(greedytmptokens, greedytmpp)] + outs

    return outs

def node2dic(node, state, tokenizer):
    d = {}
    dd = {}
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    for act, node in node.children.items():
        actstr = tokenizer.decode(act, skip_special_tokens=True)
        n = node.n
        q = node.q
        dd[actstr] = (n, q)
    d[tmpstr] = dd
    return d

def getmaxnew(step):
    '''
    return the length of token set
    '''
    if step == 0:
        return 1
    if step == 1:
        return 2
    if step == 2:
        return 4
    return 10

@torch.no_grad()
def search(root, state, model, tokenizer, dic, dicp, maxlen=1024):
    state = copy.deepcopy(state)
    cnode = root
    reward = 0
    action = (-1,-1)

    while not cnode.isleaf():
        addflag = cnode.checkadd()
        if addflag:
            maxnew = getmaxnew(cnode.step)
            agp = group_getp(state, model, dicp, topk=2, maxnew=maxnew)
            cnode.add(agp)
        action, cnode = cnode.select()
        state.extend(action)
    
    tmpstr = tokenizer.decode(state, skip_special_tokens=True)
    inds = find_all_indices(tmpstr, 'Human:')
    # check whether the generation is finished
    if len(state) > maxlen or action == terminators[0] or action == terminators[1] or terminators[0] in action or terminators[1] in action or len(inds) > 1 + 4:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
    else:
        v, embeding_token, path_n = getv(state, model, tokenizer, dic, dicp, maxlen)
        maxnew = getmaxnew(cnode.step)
        if maxnew == 1:
            gp, egp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
        else:
            gp = group_getp(state, model, dicp, topk=10, maxnew=maxnew)
            egp = copy.deepcopy(gp)

        p = [i[1] for i in gp]
        act = [i[0] for i in gp]
        acp = np.array(p)
        acp = acp / acp.sum()

        if cnode.parent == None:
            acp = 0.75 * acp + 0.25 * np.ones(len(acp)) / len(acp)
            acp = acp / acp.sum()
        acp = zip(act, acp)
        cnode.expand(root=root, ac_p=acp, reward=reward, state=state, logits=egp)
    cnode.backup(v, embeding_token, tokenizer, encoder, path_n=path_n)


@torch.no_grad()
def gmeval(batch, model, tokenizer):
    '''
    outer loop
    '''
    dic, dicp = {}, {}
    query = batch['query']
    # messages   = [
    #     # {"role": "system", "content": system_prompt},
    #     {"role": "user",   "content": query}
    # ]
    messages = query

    if query in qs:
        return None
    
    instr = messages
    # print(instr)
    input_ids = tokenizer.apply_chat_template(instr, add_generation_prompt=True, return_tensors="pt")
    # input_ids = tokenizer(instrcot, return_tensors="pt").input_ids
    slen = input_ids.shape[1]
    state = input_ids.tolist()[0]

    root = node(root=None, parent=None, prior_p=0, step=0)

    initi = 0
    while 1:
        for i in range(initi, max(maxT, initi + 15)):
            search(root, state, model, tokenizer, dic, dicp, maxlen=maxlen)
            try:
                bq, bfn = root.get_max_nq_value()
            except:
                bq, bfn = 0, 0
            if bfn > minT and bq > Vt:
                break
        act_visits = [(act, node.n) for act, node in root.children.items()]
        try:
            acts, visits = zip(*act_visits)
            visits = np.array(visits)
            targetact_probs = (visits) / (visits.sum())
            visits = visits
            act_probs = (visits) / (visits.sum())
            move = acts[int(torch.tensor(act_probs).max(dim=0).indices)]
            move = root.get_max_n_action()
            rootd = node2dic(root, state, tokenizer)
            save_dict(rootd, f'{outdir}_dicv/rain_{iterator_obj.start}-{iterator_obj.stop}.json')

            state.extend(move)
            oroot = root
            root = root.children[move]
            root.parent = None
            root.minqn = oroot.minqn
            root.maxqn = oroot.maxqn
            cp = [root.children[i].p for i in root.children]
            cp = np.array(cp)
            cp = 0.75 * cp + 0.25 * np.ones(len(cp)) / len(cp)
            cp = cp / cp.sum()
            for id, i in enumerate(root.children):
                root.children[i].p = cp[id]
            initi = root.fn
        except:
            move = tokenizer.eos_token_id

        tmpstr = tokenizer.decode(state, skip_special_tokens=True)
        inds = find_all_indices(tmpstr, 'Human:')

        if len(inds) > 1 + 4:
            break
        if len(state) > maxlen:
            break
        if move == terminators[0] or move == terminators[1]:
            break
        if terminators[0] in move or terminators[1] in move:
            break

    raina = tokenizer.decode(state, skip_special_tokens=True)

    tmp = {'question': query, 'raina': raina}  # ,'source':source, 'pa': pa,
    save_dict(dic, f'{outdir}_dicv/rain_{iterator_obj.start}-{iterator_obj.stop}.json')
    return tmp


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|end_of_text|>")
]

def increment():
    global call
    call += 1

for i in tqdm(range(len(testdataset))):
    call = 0
    start = time.time()
    print(testdataset[i])
    tmp= gmeval(testdataset[i], model, tokenizer)
    if tmp is not None:
        res.append(tmp)
        elapsed = time.time() - start
        tmp['time'] = elapsed
        tmp['call'] = call
        # {'question': query, 'raina': raina, 'best_answer':best_answer, 'time':elapsed, 'call': call}
    print('-------------------------finish one-------------------------')
    with open(f'{outdir}/rain_{iterator_obj.start}-{iterator_obj.stop}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(res, ensure_ascii=False))