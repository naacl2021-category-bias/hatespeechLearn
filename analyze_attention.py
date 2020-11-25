#!/usr/bin/env python3

import sys
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *
import pickle
from transformers import AutoTokenizer
import heapq

def get_args(parser):
  parser.add_argument("--name", type=str, default='combined-1122-02-bak')
  # combined folder name
  parser.add_argument("--batch_sz", type=int, default=32)
  parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
  parser.add_argument("--data_path", type=str, default="/")
  # test dataset
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--lr_factor", type=float, default=0.5)
  parser.add_argument("--lr_patience", type=int, default=2)
  parser.add_argument("--max_epochs", type=int, default=20)
  parser.add_argument("--n_workers", type=int, default=12)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--savedir", type=str, default="/")
  # combiend model
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--weight_classes", type=int, default=1)
  parser.add_argument("--model", type=str, default='hate_bert')
  parser.add_argument("--max_len", type = int, default = 128)
  parser.add_argument("--train", type = bool, default = False)
  parser.add_argument("--phase", type = str, default = 'test')
  parser.add_argument("--test_category", type = str, default = 'all')
  parser.add_argument("--sample_weights", type = list, default = [5997, 2157]) #label0, label1


keywords = []

def analyze(args, atts, tgt, out, text):
  bsz = tgt.size(0)
  cum = torch.zeros_like(atts[0])
  for i in range(len(atts)):
    cum += atts[i]

  cum = torch.mean(cum, 1) 

  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  tokens = []

  for i in range(bsz):
    tokens.append(tokenizer.tokenize(tokenizer.decode(text[i, :])) )

  for i in range(bsz):
    att_weights = cum[i, :, :]
    words = tokens[i]

    #import IPython; IPython.embed(); exit(1)
    text_pieces = tokenizer.decode(text[i, :]).split()

    avg_weights = att_weights.mean(0) # dim 0

    effective_att_weights = []
    effective_words = []


    for j in range(args.max_len):
      if(j < len(words)):
        if words[j] not in ['[CLS]','[SEP]','[PAD]']:
          effective_att_weights.append(avg_weights[j])
          effective_words.append(words[j])

    indices = heapq.nlargest(3, range(len(effective_att_weights)), np.array(effective_att_weights).take)
    for ind in indices:
      if tgt[i] == 1:

        effective_word = effective_words[ind].replace('##', '')
        for piece in text_pieces:
          if effective_word in piece:
            print(piece)
            keywords.append(piece)

        print(effective_att_weights[ind])
        print('-' * 20)

  #import IPython; IPython.embed(); exit(1)

def get_criterion(args):
  criterion = nn.CrossEntropyLoss()
  return criterion


def model_eval(i_epoch, data, model, args, criterion, store_preds = False):
  with torch.no_grad():
    losses, preds, tgts, raw_preds = [], [], [], []
    for batch in data:
      loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
      losses.append(loss.item())

      pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
      #import IPython; IPython.embed(); exit(1)
      raw_pred = torch.nn.functional.softmax(out, dim = 1)[:, 0].cpu().detach().numpy()

      preds.append(pred)
      raw_preds.append(raw_pred)
      tgt = tgt.cpu().detach().numpy()
      tgts.append(tgt)

  metrics = {"loss": np.mean(losses)}

  tgts = [l for sl in tgts for l in sl]
  preds = [l for sl in preds for l in sl]
  raw_preds = [l for sl in raw_preds for l in sl]

  metrics["acc"] = accuracy_score(tgts, preds)
  metrics["f1"] = f1_score(tgts, preds)
  metrics["precision"] = precision_score(tgts, preds)
  metrics["recall"] = recall_score(tgts, preds)

  if store_preds:
    store_preds_to_disk(tgts, preds, args)

  return metrics


def model_forward(i_epoch, model, args, criterion, batch):
  text, attention_mask, tgt = batch

  text, attention_mask, tgt = text.cuda(), attention_mask.cuda(), tgt.cuda()
  out, atts = model(text, attention_mask)
  #import IPython; IPython.embed(); exit(1)
  analyze(args, atts, tgt, out, text)
  loss = criterion(out, tgt)
  return loss, out, tgt


def test(args):

  set_seed(args.seed)

  train_loader, val_loader, test_loader = get_data_loaders(args)

  model = get_model(args)
  criterion = get_criterion(args)

  logger = create_logger('%s/logfile.log' % args.savedir, args)
  logger.info(model)
  model.cuda()

  start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

  if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
    checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
    start_epoch = checkpoint["epoch"]
    n_no_improve = checkpoint["n_no_improve"]
    best_metric = checkpoint["best_metric"]
    model.load_state_dict(checkpoint["state_dict"])

  # Test best model
  load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
  model.eval()
  test_metrics = model_eval(
    np.inf, test_loader, model, args, criterion, store_preds=True
  )
  log_metrics(f"Test - test", test_metrics, args, logger)


def cli_main(category):
  parser = argparse.ArgumentParser(description = 'Train Models')
  get_args(parser)
  args, remaining_args = parser.parse_known_args()
  assert remaining_args == [], remaining_args
  test(args)

  with open('encaseh2020-nonEqualAtt-' + category + '.txt', 'w') as filehandle:
    for listitem in keywords:
      filehandle.write('%s\n' % listitem)



def main():
  import warnings

  warnings.filterwarnings("ignore")

  category = sys.argv[1:][1]

  cli_main(category)




if __name__ == "__main__":

  main()


