import pprint
from typing import List, Tuple

import pyrallis
import torch
from PIL import Image

from config import RunConfig
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy
from typing import List
import importlib
from pathlib import Path

def load_pipeline_from_config(config: RunConfig):
    module_name = config.model_name
    module = importlib.import_module(module_name)
    pipeline_class = getattr(module, 'StormPipeline')
    return pipeline_class

def load_model(config: RunConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"

    StormPipeline = load_pipeline_from_config(config)
    stable = StormPipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                        for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                        if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                        "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def unique_indices(indices: List[int]) -> List[int]:
    if not indices:
        return []
    return [current for current, next_val in zip(indices, indices[1:] + [None])
        if next_val is None or current + 1 != next_val]


def fill_adj_indices(noun_indices: List[int], adj_indices: List[int]) -> Tuple[List[int], List[int]]:
    result_adj_indices: List[int | None] = [None] * len(noun_indices)

    if not adj_indices:
        return noun_indices, result_adj_indices

    adj_tensor = torch.tensor(adj_indices)
    noun_tensor = torch.tensor(noun_indices)
    differences = noun_tensor.unsqueeze(1) - adj_tensor.unsqueeze(0)

    for i, adj_idx_val in enumerate(adj_indices):
        valid_diffs = differences[:, i]
        positive_diff_mask = valid_diffs > 0
        if positive_diff_mask.any():
            closest_noun_pos_in_noun_tensor = torch.nonzero(positive_diff_mask).squeeze(1)[0].item()
            result_adj_indices[closest_noun_pos_in_noun_tensor] = adj_idx_val

    return noun_indices, result_adj_indices



def get_noun_indices_to_alter(stable, prompt: str):
    tokens = stable.tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: stable.tokenizer.decode(t).strip() for idx, t in enumerate(tokens) if 0 < idx < len(tokens) - 1}
    pprint.pprint(token_idx_to_word)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)

    nouns = {token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.text not in ['left', 'right']}
    adjs = {token.text for token in doc if token.pos_ in ['ADJ']}

    noun_indices = [idx for idx, word in token_idx_to_word.items() if word in nouns]
    adj_indices = [idx for idx, word in token_idx_to_word.items() if word in adjs]

    unique_noun_indices = unique_indices(noun_indices)
    unique_adj_indices = unique_indices(adj_indices)

    token_noun_indices, token_adj_indices = fill_adj_indices(unique_noun_indices, unique_adj_indices)
    return token_noun_indices, token_adj_indices
            

def run_on_prompt(prompt: List[str],
                model,
                controller: AttentionStore,
                token_indices: List[int],
                seed: torch.Generator,
                config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image
    
@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    prompt_list = ['a cake to the left of a suitcase']
    
    for prompt in prompt_list:
        token_indices = get_noun_indices_to_alter(stable, prompt)
        print(token_indices)
        for seed in config.seeds:
            images = []
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=prompt,
                                model=stable,
                                controller=controller,
                                token_indices=token_indices,
                                seed=g,
                                config=config)
            prompt_output_path = config.output_path / f'{prompt}'
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            image.save(prompt_output_path / f'{seed}.png')
            images.append(image)

if __name__ == '__main__':
    main()
