from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm

# import the huggingface transformers libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification

#### auto size stuff
import numpy as np

from comet import download_model, load_from_checkpoint

def increment(call):
    call += 1

def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]
###

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

# From huggingface
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
    def __init__(self, llm_path, rm_path, llm_dev="cuda:0", rm_dev="cuda:1", torch_dtype=torch.bfloat16): #torch_dtype=torch.float16
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        # if torch.cuda.device_count() > 1:
        #     self.LLM = torch.nn.DataParallel(self.LLM)
        #     self.LLM.to(self.llm_dev)
        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        
        print("Loading RM...")
        xcomet_model_path = download_model("Unbabel/wmt22-comet-da")
        self.RM = load_from_checkpoint(xcomet_model_path).to(self.rm_dev)
        # self.RM = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1, torch_dtype=torch_dtype).to(self.rm_dev)
        # self.RM.eval()
    
    def calculate_xcomet_score(self, data):
        # data = [{'src':src,'mt':answer,'ref':ref} for answer in answers]
        model_output = self.RM.predict(data, batch_size=5, gpus=1, devices=[1])
        scores = [model_output[0][i] for i in range(len(data))]
        return scores
        
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.llm_dev)
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor, input_ids) -> List[str]:
        responses = [self.tokenizer.decode(tokens[i][input_ids.shape[-1]:], skip_special_tokens=True) for i in range(len(tokens))]
        return responses
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    # def xcomet(self, mout, input_ids, src, ref):
    #     raw_tokens = input_ids[0].detach().cpu().numpy().tolist()
    #     tokens_text = search.tokens_to_text(input_ids)[0]
    #     return current_best_tokens

    def generate_greedy_step_large(self, initial_len, src, ref, mout, input_ids, pre_screen_beam_width=10, weight=0., rm_cached=None, chunk_size=1, debug=True, _use_cache=True):
        out_logits = mout.logits[:, -1] # out_logits=能夠生成的token的分數

        # top-k 分數及其 index
        # prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)
        prescreen_logits, prescreen_tokens = torch.topk(out_logits.to(self.rm_dev), dim=-1, k=pre_screen_beam_width)
        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1).to(self.rm_dev)

        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens)) # [1,k,seq_len+1(new token)]
        if debug: print(f"{to_rm_eval.shape=}")

        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1) # out_logits.shape[0] = 1 = batch_size
        if debug: print(f"{flat_trme.shape=}")
        
        # new_rm_cached = None
        current_best_score = None
        current_best_tokens = None
        if debug: print(f"{prescreen_logits.flatten().shape=}")
        for chunk, chunk_logits in zip(even_chunk(flat_trme.to(self.rm_dev), chunk_size), even_chunk(prescreen_logits.flatten(), chunk_size)):
            # pkv = None if not _use_cache else rm_cached
            decoded_texts = [self.tokenizer.decode(c[initial_len:], skip_special_tokens=True) for c in chunk]
            # decoded_texts = [self.tokenizer.decode(c, skip_special_tokens=True) for c in chunk]
            # print(decoded_texts[0])
            data = [{'src':src, 'mt':decoded_text, 'ref':ref} for decoded_text in decoded_texts]
            rewards = self.calculate_xcomet_score(data)

            # rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=chunk, attention_mask=create_attention_mask(chunk.shape[1], chunk.shape[0]).to(self.rm_dev), past_key_values=pkv, use_cache=True))
            # current_rm_cached = rm_out.past_key_values # [32,2,5,32,45,128]
            # rewards = rm_out.logits.flatten().to(self.llm_dev) #[5]
            # del rm_out
            if debug: print(f"{rewards=}")
            if debug: print(f"{rewards.shape=}")
            new_scores = torch.tensor(rewards, device=chunk_logits.device,dtype=torch.bfloat16) * weight + chunk_logits
            if debug: print(f"{new_scores=}")
            
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            current_score = new_scores[top_k_ids[0]].item()
            if debug: print(f"{current_score=} {current_best_score=} ")
            if (current_best_score is None) or (current_score > current_best_score):
                if debug: print(f"Updated!!")
                
                current_best_score = current_score
                current_best_tokens = chunk.to(self.rm_dev)[top_k_ids]
                # new_rm_cached = self.LLM._reorder_cache(current_rm_cached, top_k_ids.repeat(chunk_size,))
                del chunk  # 删除不再需要的chunk
                torch.cuda.empty_cache()
            
        if debug: print(f"{new_scores.shape=}")
        
        return current_best_tokens.to(self.llm_dev)#, new_rm_cached
        
    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")

        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")

        if rm_cached is None:
            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm_dev), past_key_values=None, use_cache=True))
            rm_cached = rm_out.past_key_values
        else:
            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=flat_trme.to(self.rm_dev), attention_mask=create_attention_mask(flat_trme.shape[1], flat_trme.shape[0]).to(self.rm_dev), past_key_values=rm_cached, use_cache=True))
            rm_cached = rm_out.past_key_values

        if debug: print(f"{rm_out.logits.flatten()=}")

        rewards = rm_out.logits.flatten().to(self.llm_dev)
        del rm_out
        if debug: print(f"{rewards.shape=}")

        new_scores = rewards * weight + prescreen_logits.flatten()
        if debug: print(f"{new_scores.shape=}")

        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
        elif method == "topk":
            # assume B=1
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
        else:
            raise ValueError(f"Invalid method '{method}'")
            
        if debug: print(f"{top_k_ids.shape=}")
        rm_cached = self.LLM._reorder_cache(rm_cached, top_k_ids.repeat(pre_screen_beam_width,))
        if debug: print(f"{rewards[top_k_ids]=}")

        return flat_trme[top_k_ids], rm_cached
    
    def generate(self, src, ref, prompt, weight=0., topk=10, max_new_token=128, method="greedy", temperature=0.7, chunk_size=1, debug=False):
        call = 0
        tokens = self.get_input_ids(prompt)
        # tokens = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.llm_dev) #shape: [1,212]

        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # [128009, 128009]
        # outputs = self.LLM.generate(tokens, max_new_tokens=40960, eos_token_id=terminators, do_sample=True, temperature=0.5, top_p=1.0, num_return_sequences=1)
        # responses = [self.tokenizer.decode(outputs[i][input_ids.shape[-1]:], skip_special_tokens=True) for i in range(len(outputs))]

        initial_len = tokens.shape[-1] #212
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            print(f"auto {chunk_size=}, {topk=}, {initial_len=}!")
        
        # if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
        #     print("The sequence of tokens is too long!!! Returning none!")
        #     return None
        
        # if tokens.shape[-1] > self.RM.config.to_dict().get("max_sequence_length", 2048):
        #     print("The sequence of tokens is too long!!! Returning none!")
        #     return None
          
        rm_cached = None
        cached = None
        
        # iterator_obj = range(initial_len)
        iterator_obj = range(max_new_token)
        if debug: iterator_obj = tqdm(iterator_obj)
        for _ in tqdm(iterator_obj):
            print(f'{iterator_obj=}')
            if debug: print(f"{type(cached)=}")
            if debug: print(f"{type(rm_cached)=}")
            with torch.no_grad():
                # if cached is None:
                mout = self.LLM(input_ids=tokens, past_key_values=None, use_cache=True)
                increment(call)
                # mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=None, use_cache=True))
                # cached = mout.past_key_values
                # else:
                #     mout = self.LLM(input_ids=tokens, past_key_values=cached, use_cache=True)
                #     # mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=cached, use_cache=True))
                #     cached = mout.past_key_values
                
                if method == "greedy_large":
                    if debug: print("large")
                    tokens = self.generate_greedy_step_large(initial_len, src, ref, mout, tokens, topk, weight, rm_cached, chunk_size, debug)
                    # print(f'{tokens.shape=}')
                    last_token_id = tokens.squeeze()[-1]
                    terminators_tensor = torch.tensor(terminators, device=last_token_id.device)
                    if last_token_id in terminators_tensor:
                        print("EOS token detected, stopping generation.")
                        del mout
                        break
                    # tokens, rm_cached = self.generate_greedy_step_large(src, ref, mout, tokens, topk, weight, rm_cached, chunk_size, debug)
                else:
                    tokens, rm_cached = self.generate_step(mout, tokens, topk, weight, method, temperature, rm_cached, debug)
                del mout

        return tokens, call
