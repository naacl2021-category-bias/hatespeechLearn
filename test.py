#!/usr/bin/env python3

import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *

def get_args(parser):
  parser.add_argument("--name", type=str, default='waseem2016')
  parser.add_argument("--batch_sz", type=int, default = 32)
  parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
  parser.add_argument("--data_path", type=str, default="/")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--lr_factor", type=float, default=0.5)
  parser.add_argument("--lr_patience", type=int, default=2)
  parser.add_argument("--max_epochs", type=int, default=50)
  parser.add_argument("--n_workers", type=int, default=12)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--savedir", type=str, default="/")
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--weight_classes", type=int, default=1)
  parser.add_argument("--model", type=str, default='hate_bert')
  parser.add_argument("--max_len", type = int, default = 128)
  parser.add_argument("--train", type = bool, default = True)
  parser.add_argument("--phase", type = str, default = 'test')
  parser.add_argument("--test_category", type = str, default = 'all')
  parser.add_argument("--sample_weights", type = list, default = [5997, 2157]) #label0, label1


def get_criterion(args):
  criterion = nn.CrossEntropyLoss()
  return criterion

def get_optimizer(model, args):

  optimizer = optim.AdamW(
    model.parameters(),
    lr = args.lr,
  )

  return optimizer


def get_scheduler(optimizer, args):
  return optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', patience = args.lr_patience, verbose = True, factor = args.lr_factor
  )


def model_eval(i_epoch, data, model, args, criterion, store_preds = False):
  with torch.no_grad():
    losses, preds, tgts = [], [], []
    for batch in data:
      loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
      losses.append(loss.item())

      pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

      preds.append(pred)
      tgt = tgt.cpu().detach().numpy()
      tgts.append(tgt)

  metrics = {"loss": np.mean(losses)}

  tgts = [l for sl in tgts for l in sl]
  preds = [l for sl in preds for l in sl]
  metrics["acc"] = accuracy_score(tgts, preds)
  
  metrics["precision"] = precision_score(tgts, preds)
  metrics["recall"] = recall_score(tgts, preds)
  metrics["f1"] = f1_score(tgts, preds)

  if store_preds:
    store_preds_to_disk(tgts, preds, args)

  return metrics


def model_forward(i_epoch, model, args, criterion, batch):
  text, attention_mask, tgt = batch

  text, attention_mask, tgt = text.cuda(), attention_mask.cuda(), tgt.cuda()
  out, atts = model(text, attention_mask)

  loss = criterion(out, tgt)
  return loss, out, tgt


def test(args):

  set_seed(args.seed)
  args.savedir = os.path.join(args.savedir, args.name)
  #os.makedirs(args.savedir, exist_ok = True)

  train_loader, val_loader, test_loader = get_data_loaders(args)

  model = get_model(args)
  criterion = get_criterion(args)
  optimizer = get_optimizer(model, args)
  scheduler = get_scheduler(optimizer, args)

  logger = create_logger('%s/logfile.log' % args.savedir, args)
  #logger.info(model)
  model.cuda()

  #torch.save(args, os.path.join(args.savedir, 'args.pt'))

  start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

  if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
    checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
    start_epoch = checkpoint["epoch"]
    n_no_improve = checkpoint["n_no_improve"]
    best_metric = checkpoint["best_metric"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

  # Test best model
  load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
  model.eval()
  test_metrics = model_eval(
    np.inf, test_loader, model, args, criterion, store_preds=True
  )
  log_metrics(f"Test - test", test_metrics, args, logger)


def cli_main():
  parser = argparse.ArgumentParser(description = 'Train Models')
  get_args(parser)
  args, remaining_args = parser.parse_known_args()
  assert remaining_args == [], remaining_args
  test(args)


if __name__ == "__main__":
  import warnings

  warnings.filterwarnings("ignore")

  cli_main()

