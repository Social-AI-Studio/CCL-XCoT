# import argparse
# import os

# from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
# from tqdm import tqdm
# from transformers import AutoTokenizer
# import json


# def config_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default=None, 
#                         help='Directory to load the raw WikiText-103 dataset from')
#     parser.add_argument('--output_dir', type=str, default=None, 
#                         help='Directory to store preprocessed dataset')
#     parser.add_argument('--seq_length', type=int, default=512, 
#                         help='Maximum sequence length while tokenizing')
#     parser.add_argument('--chars_per_tok', type=float, default=3.2, 
#                         help='Number of characters per token, on average')
#     parser.add_argument('--pad_token_id', type=int, default=0, 
#                         help='Padding token ID')
#     parser.add_argument("--model_name", type=str, default='google/gemma-7b', 
#                         choices=["gpt2"])
#     args = parser.parse_args()
#     args.max_chars_per_seq = args.seq_length * args.chars_per_tok
#     return args


# def truncate_and_tokenize(args, tokenizer, text_data):
#     ret_dict = {
#         'content': [],
#         'input_ids': [],
#         'attention_mask': [],
#         'data_idx': [],
#     }
#     for data_idx, txt in tqdm(enumerate(text_data), total=len(text_data)):
#         tmp_content=[]
#         tmp_input_ids=[]
#         tmp_attention_mask=[]
#         eos_token=tokenizer.eos_token
#         for t in txt['text']:
#             t = t+eos_token
#             features = tokenizer(t,
#                                 padding='max_length',
#                                 truncation=True,
#                                 max_length=args.seq_length,
#                                 add_special_tokens=True)
#             tmp_content.append(t)
#             tmp_input_ids.append(features['input_ids'])
#             tmp_attention_mask.append(features['attention_mask'])
#         ret_dict['content'] += [tmp_content]
#         ret_dict['input_ids'] += [tmp_input_ids]
#         ret_dict['attention_mask'] += [tmp_attention_mask]
#         ret_dict['data_idx'].append(data_idx)
#         # import pdb;pdb.set_trace()
#     print(len(text_data), ret_dict.keys(), 
#           len(ret_dict['content']), len(ret_dict['input_ids']))
#     return ret_dict


# def tokenize_wikitext_103(args):
#         output_path = "../Data/processed_train"
#         data_path = "/home/weihua/Haullucination/ContraCLM/Data/Training_Data/Data/Training.json"
#         os.makedirs(output_path, exist_ok=True)
#         tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#         tokenizer.padding_side = 'right'
#         with open(data_path, "r", encoding="utf-8") as f:
#             raw_data = json.load(f)
#         text_data = [item["text"] for item in raw_data]
#         dataset = Dataset.from_dict({"text": text_data})
#         tokenized_dict = truncate_and_tokenize(args, tokenizer, dataset)
#         tokenized_dataset = Dataset.from_dict(tokenized_dict)
#         tokenized_dataset.save_to_disk(output_path)



# def merge_data(args):
#     datadict = {}
#     for datatype in ["validation", "test", "train"]:
#         data_path = os.path.join(args.output_dir, f"wikitext103_raw_v1_{datatype}")
#         dataset = load_from_disk(data_path)
#         print(type(dataset), len(dataset))
#         if datatype == "validation":
#             datadict["valid"] = dataset
#         else:
#             datadict[datatype] = dataset
#     merged_dataset = DatasetDict(datadict)
#     os.makedirs(args.output_dir, exist_ok=True)
#     merged_dataset.save_to_disk(
#         os.path.join(args.output_dir, f"wikitext103_raw_v1_tokenized_all")
#     )


# if __name__ == "__main__":
#     args = config_args()
#     tokenize_wikitext_103(args)

import argparse
import os
import json

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def config_args():
    parser = argparse.ArgumentParser()
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument('--data_path', type=str,
                    help='Path to a single input JSON file (list of items with "text" or raw strings).')
    mx.add_argument('--data_dir', type=str,
                    help='Directory containing Train.json and Valid.json.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store preprocessed dataset')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='Maximum sequence length while tokenizing')
    parser.add_argument('--chars_per_tok', type=float, default=3.2,
                        help='Number of characters per token, on average')
    parser.add_argument('--pad_token_id', type=int, default=None,
                        help='Padding token ID (optional). If not set, will fall back to tokenizer.pad_token_id or eos_token_id.')
    parser.add_argument("--model_name", type=str, default='gpt2',
                        help='HF model name for tokenizer (e.g., "gpt2", "google/gemma-7b")')

    args = parser.parse_args()
    args.max_chars_per_seq = args.seq_length * args.chars_per_tok
    return args


def _read_json_as_text_list(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise ValueError(f"Input JSON should be a list. Got type={type(raw_data)} for {path}")

    if len(raw_data) > 0 and isinstance(raw_data[0], dict) and "text" in raw_data[0]:
        text_data = [item["text"] for item in raw_data]
    else:
        text_data = [str(item) for item in raw_data]
    return text_data


def truncate_and_tokenize(args, tokenizer, dataset):
    """
    dataset: a HuggingFace Dataset with a 'text' column, where each row is a string.
    """
    ret_dict = {
        'content': [],
        'input_ids': [],
        'attention_mask': [],
        'data_idx': [],
    }

    eos_token = tokenizer.eos_token or ""
    use_pad_id = (
        args.pad_token_id
        if args.pad_token_id is not None
        else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    )

    # 设置 pad token（若需要）
    if tokenizer.pad_token_id is None and use_pad_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = use_pad_id

    for data_idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        text = row['text']
        t = text + eos_token if eos_token else text
        features = tokenizer(
            t,
            padding='max_length',
            truncation=True,
            max_length=args.seq_length,
            add_special_tokens=True
        )

        ret_dict['content'].append(t)
        ret_dict['input_ids'].append(features['input_ids'])
        ret_dict['attention_mask'].append(features['attention_mask'])
        ret_dict['data_idx'].append(data_idx)

    print(len(dataset), ret_dict.keys(), len(ret_dict['content']), len(ret_dict['input_ids']))
    return ret_dict


def tokenize_text_list(args, tokenizer, text_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dataset = Dataset.from_dict({"text": text_list})
    tokenized_dict = truncate_and_tokenize(args, tokenizer, dataset)
    tokenized_dataset = Dataset.from_dict(tokenized_dict)
    tokenized_dataset.save_to_disk(save_dir)
    print(f"Saved tokenized dataset to {save_dir}")
    return tokenized_dataset


def process_single_file(args, tokenizer):
    text_list = _read_json_as_text_list(args.data_path)
    save_dir = args.output_dir  # 保持与你原脚本一致：单文件直接存到 output_dir
    return tokenize_text_list(args, tokenizer, text_list, save_dir)


def process_data_dir(args, tokenizer):
    # 约定文件名
    split_map = {
        "train": "Train.json",
        "valid": "Valid.json",
    }

    produced = {}
    for split, fname in split_map.items():
        path = os.path.join(args.data_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected file not found: {path}")

        print(f"Processing {split} from {path}")
        text_list = _read_json_as_text_list(path)
        split_out = os.path.join(args.output_dir, split)
        produced[split] = tokenize_text_list(args, tokenizer, text_list, split_out)

    # 合并为 DatasetDict 并另存
    merged = DatasetDict({k: v for k, v in produced.items()})
    merged_dir = os.path.join(args.output_dir, "tokenized_all")
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_to_disk(merged_dir)
    print(f"Merged DatasetDict saved to {merged_dir}")


def main():
    args = config_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'right'

    if args.data_dir:
        process_data_dir(args, tokenizer)
    else:
        process_single_file(args, tokenizer)


if __name__ == "__main__":
    main()

