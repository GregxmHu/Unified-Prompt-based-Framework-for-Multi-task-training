############################################################################################################
from transformers import T5ForConditionalGeneration, T5Tokenizer
from mnli_dataloader import MNLIDataLoader
from mnli_dataset import MNLIDataset
from mnli_model import MNLIT5
############################################################################################################

import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import AdamW, Adafactor

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import is_first_worker, DistributedEvalSampler,set_dist_args, optimizer_to
from contextlib import nullcontext # from contextlib import suppress as nullcontext # for python < 3.7
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import random
import numpy as np
from tqdm import tqdm
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# writer = SummaryWriter(log_dir='logs')


def dev(args, model, dev_loader, device, tokenizer):
    total=0
    right=0
    for dev_batch in tqdm(dev_loader, disable=args.local_rank not in [-1, 0]):
        #query_id, doc_id, label= dev_batch[''], dev_batch['doc_id'], dev_batch['label']
        with torch.no_grad():
            if args.original_t5:
                output_sequences = model.module.generate(
                    input_ids=dev_batch['input_ids'].to(device),
                    attention_mask=dev_batch['attention_mask'].to(device),
                    do_sample=False, # disable sampling to test if batching affects output
                )
                batch_result = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                # print(batch_result)
                true_result = dev_batch["raw_label"]
                total += len(true_result)
                for br, tr in zip(batch_result, true_result):
                    if br == tr:
                        right += 1
            else:
                batch_score = model(
                        input_ids=dev_batch['input_ids'].to(device), 
                        attention_mask=dev_batch['attention_mask'].to(device), 
                        decoder_input_ids=dev_batch['decoder_input_ids'].to(device),
                        )
                predict=torch.argmax(batch_score,dim=1)
                label=dev_batch['label'].to(device)
                total+=len(label)
                right+=torch.eq(predict,label).sum()
    # return int(total),int(right.detach().cpu())
    return total, right


def batch_to_device(batch, device):
    device_batch = {}
    for key, value in batch.items():
        device_batch[key] = value.to(device)
    return device_batch
            

def train(args, model, loss_fn, m_optim, m_scheduler, train_loader, dev_loader, device, train_sampler=None, tokenizer=None):
    best_mes = 0.0
    global_step = 0 # steps that outside epoches
    force_break = False
    for epoch in range(args.epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed
            logger.warning("current gpu local_rank {}".format(args.local_rank))

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            # print("Before: global step {}, rank {}".format(global_step, args.local_rank))
            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext
            if args.original_t5:
                with sync_context():
                    batch_loss = model(                        
                        input_ids=train_batch['input_ids'].to(device), 
                        attention_mask=train_batch['attention_mask'].to(device), 
                        labels=train_batch["labels"].to(device)
                    ).loss
            else:
                with sync_context():
                    batch_score = model(
                        input_ids=train_batch['input_ids'].to(device), 
                        attention_mask=train_batch['attention_mask'].to(device), 
                        decoder_input_ids=train_batch['decoder_input_ids'].to(device),
                        )
                with sync_context():
                    batch_loss = loss_fn(batch_score, train_batch['label'].to(device))

            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            with sync_context():
                batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                m_optim.step()
                if m_scheduler is not None:
                    m_scheduler.step()
                m_optim.zero_grad()
              

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    
                    if args.local_rank in [-1, 0]:
                        logger.info("training gpu {}:,  global step: {}, local step: {}, loss: {}".format(args.local_rank,global_step+1, step+1, avg_loss/args.logging_step))
                        if args.tb is not None:
                            args.tb.add_scalar("loss", avg_loss/args.logging_step, global_step + 1)
                            args.tb.add_scalar("epochs", epoch + 1, global_step + 1)
                    avg_loss = 0.0 

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):
                    model.eval()
                    with torch.no_grad():
                        total,right=dev(args, model, dev_loader, device, tokenizer)
                    model.train()

                    if args.local_rank != -1:
                        with open(args.res,'a+') as f:
                            f.write(json.dumps({'total':total,'right':right}))
                            f.write('\n')
                        dist.barrier()
                    if args.local_rank in [-1,0]:
                        with open(args.res,'r') as f:
                            r,t=0,0
                            for line in f:
                                r+=eval(line)['right']
                                t+=eval(line)['total']
                                print(r,t)
                            mes=r/t
                        os.remove(args.res)
                        best_mes = mes if mes >= best_mes else best_mes
                        logger.info("global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))
                
                global_step += 1

                if args.max_steps is not None and global_step == args.max_steps:
                    force_break = True
                    break
            # print("After: global step {}, rank {}".format(global_step, args.local_rank))
            if args.local_rank != -1:
                dist.barrier()
            # print("After barrier: global step {}, rank {}".format(global_step, args.local_rank))
        # print("After epoch: global step {}, rank {}".format(global_step, args.local_rank))
        if args.local_rank != -1:
            dist.barrier()
        # print("After epoch barrier: global step {}, rank {}".format(global_step, args.local_rank))
        if force_break:
            break
    if args.local_rank != -1:
        dist.barrier()
    # print("Finish training. {}".format(args.local_rank))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-train', type=str, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', type=str, default='./data/dev_toy.jsonl')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float,help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('-n_warmup_steps',type=int,default=0)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-right',type=int,default=0)
    parser.add_argument("--original_t5", action="store_true")
    args = parser.parse_args()
    set_dist_args(args) # get local cpu/gpu device
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
        args.tb = writer
    else:
        args.tb = None

    tokenizer = T5Tokenizer.from_pretrained(args.vocab)
    logger.info('reading training data...')
    train_set=MNLIDataset(dataset=args.train,tokenizer=tokenizer)
    logger.info('reading dev data...')
    dev_set=MNLIDataset(dataset=args.dev,tokenizer=tokenizer, max_input=200)

    if args.local_rank != -1:
        
        train_sampler = DistributedSampler(train_set)
        train_loader = MNLIDataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            sampler=train_sampler
        )
        dev_sampler = DistributedEvalSampler(dev_set)
        dev_loader = MNLIDataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
            shuffle=False,
            num_workers=8,
            sampler=dev_sampler
        )
        dist.barrier()

    model = MNLIT5(args.pretrain) if not args.original_t5 else T5ForConditionalGeneration.from_pretrained(args.pretrain)

    device = args.device
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    model.zero_grad()
    model.train()
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == "adafactor":
        m_optim = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    if args.optimizer.lower() == "adafactor":
        m_scheduler = None
    else:
        if args.local_rank == -1:
            m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size if args.max_steps is None else args.max_steps)
        else:
            m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.world_size*args.gradient_accumulation_steps) if args.max_steps is None else args.max_steps)

    if m_optim is not None:
        optimizer_to(m_optim,device)

    logger.info(args)
    train(args, model, loss_fn, m_optim, m_scheduler,  train_loader, dev_loader, device, train_sampler=train_sampler, tokenizer=tokenizer)
    if args.local_rank != -1:
        dist.barrier()
if __name__ == "__main__":
    main()
