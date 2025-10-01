# Data

### Obtaining the Raw Data

* The English and Chinese parallel data for Curriculum-based Contrastive Learning is obtaining from WMT 2024(https://www2.statmt.org/wmt24/).
* The first stage training we use 100K parallel data from sentence-level MT task (https://www2.statmt.org/wmt24/).
* The second stage training need to use 10K document-level training data from https://www2.statmt.org/wmt24/translation-task.html#_document_level_mt.

### Preprocessing

This folder contains scripts to preprocess the datasets. Assuming access to raw text data, our multi-stage preprocessing follows these steps:


#### 1. Generating Splits

Each row in the dataset is shuffled and split into train/valid/test spilts. The ratios of the splits can be customized. Each split is stored as a `.arrow` dataset using huggingface `datasets` library.


#### 2. Chunking & Tokenization

Each raw string in each split from above is chunked up to a custom threshold. We chunk up to the last line (i.e., finding `\n`) such that the resulting chunk has <= `max_chars_per_token * sequence_length` characters. This chunk is tokenized and the resulting dataset is stored as a `.arrow` dataset using huggingface `datasets` library.

#### 3. Exact Match (EM) Deduplication

After chunking, the resulting datasets may have duplicate strings. To eliminate these, we simply drop duplicate strings based on EM. The summary of this step is `output_data_strs := set(input_data_strs)`.

## Usage

```

* For natural language data, set the relevant paths and variables in `run_preprocess_wikitext.sh` and run:
```shell
bash run_preprocess_CCL_Data.sh
```

#### References

[1] Su, Yixuan, et al. "A contrastive framework for neural text generation." arXiv preprint arXiv:2202.06417 (2022).
[2] Nihal Jain, et al. "ContraCLM: Contrastive Learning For Causal Language Model". 2023.acl-long.355
