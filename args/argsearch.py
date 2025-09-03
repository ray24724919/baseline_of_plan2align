import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from trl import AutoModelForCausalLMWithValueHead
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

# reward based search
class ARGS:
    def __init__(self, llm_path, rm_path, llm_dev="cuda:0", rm_dev="cuda:1", torch_dtype=torch.bfloat16):
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        self.LLM.gradient_checkpointing_enable()
        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        print("Loading RM...")
        self.rm_path = rm_path
        self.RM = AutoModelForCausalLMWithValueHead.from_pretrained(rm_path, torch_dtype=torch.bfloat16, device_map=self.rm_dev, trust_remote_code=True)
        value_head_file = hf_hub_download(repo_id=rm_path, filename="value_head.safetensors")
        value_head_weights = load_file(value_head_file)
        new_state_dict = {}
        for key, value in value_head_weights.items():
            if key.startswith("v_head."):
                new_key = key.replace("v_head.", "") 
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        self.RM.v_head.load_state_dict(new_state_dict)
        self.RM.eval()
        self.RM.gradient_checkpointing_enable()
    
    # def _messages_to_history(self, msgs):
    #     """
    #     把多輪 messages 轉成:
    #         User: ...\nAssistant: ...\nUser: ...
    #     不保留 system 句。
    #     """
    #     lines = []
    #     for m in msgs:
    #         if m["role"] == "system":
    #             continue
    #         tag = "User:" if m["role"] == "user" else "Assistant:"
    #         lines.append(f"{tag} {m['content']}")
    #     return "\n".join(lines)
    
    def reward_fn(self, prompt, response):
        messages = prompt + [{"role": "assistant", "content": response}]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.rm_dev)

        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs.to(self.rm_dev)}
        else:
            for k, v in inputs.items():
                inputs[k] = v.to(self.rm_dev)
        with torch.no_grad():
            outputs = self.RM(**inputs, return_value=True)
            reward = outputs[2][:, -1].item()
        return reward
    
    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, debug=False):
        last_logits = mout.logits[:, -1]
        cand_logits, cand_ids = torch.topk(last_logits, k=pre_screen_beam_width, dim=-1)

        scores, tok_list = [], []
        for logit, tok in zip(cand_logits.flatten(), cand_ids.flatten()):
            tok_str = self.tokenizer.decode(tok, skip_special_tokens=True)

            candidate_resp = self._assistant_generated + tok_str
            reward = self.reward_fn(self._history_str, candidate_resp)

            score = reward * weight + logit.item()
            scores.append(score)
            tok_list.append(tok)

        scores_t = torch.tensor(scores, device=self.llm_dev)

        if method == "greedy":
            sel_idx = torch.argmax(scores_t).item()
        elif method == "topk":
            probs = F.softmax(scores_t / temperature, dim=-1)
            sel_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            raise ValueError(f"Invalid method '{method}'")

        sel_tok_id = tok_list[sel_idx]
        sel_tok = sel_tok_id.view(1, 1).to(self.llm_dev)
        return sel_tok  
        # new_input_ids = torch.cat([input_ids, sel_tok], dim=-1)
        # return new_input_ids

    def generate_greedy_step_large(self, mout, input_ids, pre_screen_beam_width=10, weight=0., debug=True):
        last_logits = mout.logits[:, -1] # (1, vocab)
        cand_logits, cand_ids = torch.topk(last_logits, k=pre_screen_beam_width, dim=-1) #  (1, k)

        best_score, best_tok = None, None
        for logit, tok_id in zip(cand_logits.flatten(), cand_ids.flatten()):
            token_str = self.tokenizer.decode(tok_id, skip_special_tokens=True)
            candidate_resp = self._assistant_generated + token_str   
            reward = self.reward_fn(self._history_str, candidate_resp)
            score = reward * weight + logit.item()

            if (best_score is None) or (score > best_score):
                best_score, best_tok = score, tok_id

        best_tok = best_tok.view(1, 1).to(self.llm_dev)
        # new_ids = torch.cat([input_ids, best_tok], dim=-1)
        return best_tok    
        # return new_ids
    
    def generate(self, prompt, weight=0., topk=10, max_new_token=128, method="greedy", temperature=0.7, chunk_size=2, debug=False):
        call = 0
        # if isinstance(prompt, str):
        #     msg_list = [{"role": "user", "content": prompt}]
        # else:
        msg_list = prompt

        self._history_str = msg_list
        ids = self.tokenizer.apply_chat_template(msg_list, tokenize=True, add_generation_prompt=True)
        tokens = torch.tensor([ids], device=self.llm_dev)
        self._assistant_generated = ""
        
        terminators = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        }
        seq_tokens  = torch.tensor([ids], device=self.llm_dev)
        step_tokens = seq_tokens
        cached      = None
        loop_iter = tqdm(range(max_new_token))
        MAX_KV = 256

        for step in loop_iter:
            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(input_ids=step_tokens, use_cache=True)
                else:
                    mout = self.LLM(input_ids=step_tokens, past_key_values=cached, use_cache=True)
                cached = mout.past_key_values
                call  += 1

            if method == "greedy_large":
                new_tok = self.generate_greedy_step_large(mout, seq_tokens, topk, weight, debug=debug)
            else:
                new_tok = self.generate_step(mout, seq_tokens, topk, weight, method, temperature, debug)
            del mout

            seq_tokens  = torch.cat([seq_tokens, new_tok], dim=-1)
            step_tokens = new_tok
            self._assistant_generated += self.tokenizer.decode(new_tok[0], skip_special_tokens=True)
            # print(self._assistant_generated)
            
            if new_tok.item() in terminators:
                break
            if step % MAX_KV == 0:
                cached = None
                step_tokens = seq_tokens
        return seq_tokens, call