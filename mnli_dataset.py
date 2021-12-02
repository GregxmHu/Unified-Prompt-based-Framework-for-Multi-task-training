from os import truncate
from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import T5Tokenizer

class MNLIDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: T5Tokenizer,
        max_input: int = 1280000,
    ) -> None:
        self._label_mapping=['entailment', 'neutral', 'contradiction']
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        with open(self._dataset,'r') as f:
            self._examples=[eval(line) for line in f]        


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        text='mnli hypothesis: ' + example["hypothesis"] + ' premise: ' + example["premise"]
        tokenized = self._tokenizer(text, padding="max_length", truncation=True, max_length=384)
        source_ids, source_mask = tokenized["input_ids"], tokenized["attention_mask"]
        tokenized = self._tokenizer(self._label_mapping[example["label"]], padding="max_length", truncation=True, max_length=10)
        target_ids = tokenized["input_ids"]
        target_ids = [
           (label if label != self._tokenizer.pad_token_id else -100) for label in target_ids
        ]
        raw_label = self._label_mapping[example["label"]]
        output = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "raw_label": raw_label
        }
        return output

    def __len__(self) -> int:
        return len(self._examples)


    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        raw_label = [item["raw_label"] for item in batch]
        return {'input_ids': input_ids, "attention_mask": attention_mask, 'labels': labels, "raw_label": raw_label}

