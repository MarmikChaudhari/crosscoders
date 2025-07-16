from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as th
from tqdm import tqdm
import pytest
from nnsight import LanguageModel
from dictionary_learning.cache import ActivationCache
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

n_samples_simple_stories = 5 # 1_000_000 
n_samples_code = 2 # 210_000 
n_samples_arxiv = 2 # 21_000 # down sample as avg sample length of arxiv is way higher than other subsets
test_split = 0.05



def get_dataset_tokenizer(n_samples_simple_stories=n_samples_simple_stories,n_samples_code=n_samples_code,n_samples_arxiv=n_samples_arxiv,test_split=test_split):
    # get the three subsets of arxiv, code and simple stories
    arxiv_url = [
        f"https://olmo-data.org/dolma-v1_7/redpajama-arxiv/arxiv-{i:04d}.json.gz" for i in range(2)
    ]
    code_url = ["https://olmo-data.org/dolma-v1_7/starcoder/starcoder-0000.json.gz"]
    simeple_stories_ds = load_dataset("SimpleStories/SimpleStories", split="train", streaming=True)

    arxiv_ds = load_dataset(
        "json",
        data_files=arxiv_url,
        split="train",
        streaming=True,
    )
    code_ds = load_dataset(
        "json",
        data_files=code_url,
        split="train",
        streaming=True,
    )

    simple_stories_data = []
    for i, item in enumerate(tqdm(simeple_stories_ds, desc="SimpleStories", total=n_samples_simple_stories)):
        if i >= n_samples_simple_stories:
            break
        simple_stories_data.append({"text": item["story"]})
    
    arxiv_data = []
    for i, item in enumerate(tqdm(arxiv_ds, desc="Arxiv", total=n_samples_arxiv)):
        if i >= n_samples_arxiv:
            break
        arxiv_data.append({"text": item["text"]})
    
    code_data = []
    for i, item in enumerate(tqdm(code_ds, desc="Code", total=n_samples_code)):
        if i >= n_samples_code:
            break
        code_data.append({"text": item["text"]})

    datasets = [
        Dataset.from_list(simple_stories_data),
        Dataset.from_list(arxiv_data),
        Dataset.from_list(code_data)
    ]

    # combine
    combined_ds = concatenate_datasets(datasets)

    print(f"total items: {len(combined_ds)}")

    return combined_ds



combined_ds = get_dataset_tokenizer()

dataset = combined_ds["text"]  # Extract text list from the Dataset object

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", device_map="auto", torch_dtype=th.float32
)

model = LanguageModel(model, torch_dtype=th.float32, tokenizer=tokenizer)
model.tokenizer.pad_token = model.tokenizer.eos_token

# get a transformer block to extract activations from
target_layer = model.transformer.h[6]  # Middle layer of GPT-2
submodule_name = "transformer_h_6"

# parameters for activation collection
batch_size = 2
context_len = 64
d_model = 768  # GPT-2 hidden size
temp_dir = "dataset_activations"

# collect activations using ActivationCache
ActivationCache.collect(
    data=dataset,
    submodules=(target_layer,),
    submodule_names=(submodule_name,),
    model=model,
    store_dir=temp_dir,
    batch_size=batch_size,
    context_len=context_len,
    shard_size=1000,  # Small shard size for testing
    d_model=d_model,
    io="out",
    max_total_tokens=10000,
    store_tokens=True,
)