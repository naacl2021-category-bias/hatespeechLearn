#!/usr/bin/env python3

import json
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed
#from transformers import BertTokenizer
from transformers import AutoTokenizer
import csv

class JsonlDataset(Dataset):
  def __init__(self, data_path, args):

    temp = [json.loads(line) for line in open(data_path)]
    self.data = []

    # put empty category for non hate
    for i in temp:
      if i['label'] == 1:
        if args.phase == 'test' and args.test_category != 'all':
          if i['category'][0] == args.test_category:
            self.data.append(i)
        else:
          self.data.append(i)
      else:
        i['category'] = []
        self.data.append(i)

    self.args = args
    self.n_classes = len(args.labels)
    #self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = True, max_len = args.max_len)
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
   

    # return self.data[index]["text"], self.data[index]["label"], self.data[index]["category"]

    return self.data[index]["text"], self.data[index]["label"]


