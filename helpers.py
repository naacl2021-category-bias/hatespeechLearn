#!/usr/bin/env python3

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import JsonlDataset
import numpy as np
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import AutoTokenizer

def get_labels_and_frequencies(path):
  label_freqs = Counter()
  data_labels = [json.loads(line)["label"] for line in open(path)]
  if type(data_labels[0]) == list:
    for label_row in data_labels:
      label_freqs.update(label_row)
  else:
    label_freqs.update(data_labels)

  return list(label_freqs.keys()), label_freqs

def collate_fn(batch, tokenizer, args):

  text = [i[0] for i in batch]
  label = [i[1] for i in batch]
  # category = [i[2] for i in batch]

  stuff = tokenizer.batch_encode_plus(text, 
          padding=True, 
          truncation=True, 
          pad_to_max_length=True, 
          max_length = args.max_len, 
          return_tensors="pt")

  text_tensor = stuff['input_ids']
  attention_mask = stuff['attention_mask']
  tgt_tensor = torch.LongTensor(label)

  #return text_tensor, attention_mask, tgt_tensor, category
  return text_tensor, attention_mask, tgt_tensor


def get_data_loaders(args):

  args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, "train.jsonl"))
  
  args.n_classes = len(args.labels)

  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  
  collate = functools.partial(collate_fn, tokenizer=tokenizer, args=args)

  train = JsonlDataset(os.path.join(args.data_path, "train.jsonl"), args)

  args.train_data_len = len(train)

  dev = JsonlDataset(os.path.join(args.data_path, "dev.jsonl"), args)

  test = JsonlDataset(os.path.join(args.data_path, "test.jsonl"), args)

  samples_weights = []
  #weights = 1. / torch.tensor([13869, 6799], dtype = torch.float)
  weights = 1. / torch.tensor(args.sample_weights, dtype = torch.float)
  for i in train:
    label = i[1]
    samples_weights.append(weights[label])

  sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement = True)
  #import IPython; IPython.embed(); exit(1)

  train_loader = DataLoader(
    train,
    batch_size=args.batch_sz,
    #shuffle=True,
    num_workers=args.n_workers,
    collate_fn=collate,
    sampler=sampler
  )

  val_loader = DataLoader(
    dev,
    batch_size=args.batch_sz,
    shuffle=False,
    num_workers=args.n_workers,
    collate_fn=collate,
  )

  test_loader = DataLoader(
    test,
    batch_size=args.batch_sz,
    shuffle=False,
    num_workers=args.n_workers,
    collate_fn=collate,
    drop_last=True
  )

  return train_loader, val_loader, test_loader
