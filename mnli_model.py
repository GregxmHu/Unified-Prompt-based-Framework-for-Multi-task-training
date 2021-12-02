from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
from torch import nn
import torch
class MNLIT5(nn.Module):
    def __init__(self,checkpoint:str):
        super(MNLIT5,self).__init__()
        self.t5=T5ForConditionalGeneration.from_pretrained(checkpoint)      
    
    def forward(self,input_ids,attention_mask,decoder_input_ids):
        output=self.t5(input_ids=input_ids,decoder_input_ids=decoder_input_ids,attention_mask=attention_mask,return_dict=True)
        logits=output.logits
        batch_score=logits[:,0,[1176,7163,6136]] 
        return batch_score

