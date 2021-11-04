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
        ckpt: str,
        max_input: int = 1280000,
    ) -> None:
        self._label_mapping=['true','neutral','false']
        #对应[1176,7163,6136]
        self._dataset = dataset
        self._tokenizer = T5Tokenizer.from_pretrained(ckpt)
        self._max_input = max_input
        with open(self._dataset,'r') as f:
            self._examples=[eval(line) for line in f]        

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        text='Premise: '+example["premise"]+' Hypothesis: '+example["hypothesis"]+' Entailment: '
        output=self._tokenizer(text,padding="max_length",truncation=True,max_length=384)
        output.update({'decoder_input_ids':[0],'label':example['label']})
        return output

    def __len__(self) -> int:
        return len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        
        decoder_input_ids = torch.tensor([item['decoder_input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        label = torch.tensor([item['label'] for item in batch])
        return {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'label': label}