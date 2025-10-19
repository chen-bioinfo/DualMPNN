import argparse
from sympy import true
from torch.utils.data import Subset
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
from concurrent.futures import ProcessPoolExecutor    
from model.utils import Struct_DualDataset, MultiStruct_Loader
from model.DualMPNN_util import template_featurize, loss_smoothed, loss_nll, get_std_opt, featurize
from model.DualMPNN_model import DualMPNN
from multiprocessing import Value, Lock
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a scheduler with cosine decay after a warmup phase """

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine Decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MyArgs(object):
  def __init__(self):
    #--------------------------------------------------------#
    self.device_set = "cuda:0"
    self.path_for_outputs = "train_outputs/test"
    #--------------------------------------------------------#
    self.previous_checkpoint = ""
    self.num_epochs = 100
    self.save_model_every_n_epochs = 5
    self.reload_data_every_n_epochs = 4
    self.num_examples_per_epoch = 20000
    self.num_examples_valid = 20000
    self.batch_size = 15000
    self.max_protein_length = 200000
    self.hidden_dim = 128
    self.num_encoder_layers = 4
    self.num_decoder_layers = 4
    self.num_neighbors = 48
    self.dropout = 0.1
    self.backbone_noise = 0.0
    self.rescut = 3.5
    self.debug = False
    self.gradient_norm = 3.0 
    self.mixed_precision= True

args = MyArgs()

scaler = torch.cuda.amp.GradScaler()

base_folder = time.strftime(args.path_for_outputs, time.localtime())

if base_folder[-1] != '/':
    base_folder += '/'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['model_weights']
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

PATH = args.previous_checkpoint

logfile = base_folder + 'log.txt'
if not PATH:
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

if args.debug:
    args.num_examples_per_epoch = 200
    args.num_examples_valid = 200
    args.max_protein_length = 1000
    args.batch_size = 200

# Load training data
pdb_dict_train = torch.load('dataset/T500/T500_train.pt')
dataset_train = Struct_DualDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
loader_train = MultiStruct_Loader(dataset_train, batch_size=args.batch_size, shuffle=False)

# Load validation data
pdb_dict_valid = torch.load('dataset/T500/Test500.pt')
dataset_valid = Struct_DualDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length) 
loader_valid = MultiStruct_Loader(dataset_valid, batch_size=args.batch_size, shuffle=False)

# Create the model
model = DualMPNN(node_features=args.hidden_dim, 
                    edge_features=args.hidden_dim, 
                    hidden_dim=args.hidden_dim, 
                    num_encoder_layers=args.num_encoder_layers, 
                    num_decoder_layers=args.num_encoder_layers, 
                    k_neighbors=args.num_neighbors, 
                    dropout=args.dropout, 
                    augment_eps=args.backbone_noise)

# Move the model to GPU
device = torch.device(args.device_set if torch.cuda.is_available() else "cpu")
model.to(device)

if PATH:
    print(f"Load checkpoint:  {PATH}")
    checkpoint = torch.load(PATH)
    total_step = checkpoint['step']  # Load total_step from the checkpoint
    epoch = checkpoint['epoch']  # Load epoch from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    total_step = 0
    epoch = 0

optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler = get_cosine_schedule_with_warmup(optimizer, 4000, len(loader_train)*args.num_epochs)
if PATH:
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

train_len, valid_len, test_len = 0, 0, 0
for batch in loader_train:
    for b in batch:
        train_len += 1
for batch in loader_valid:
    for b in batch:
        valid_len += 1
print(f"Train data size: {train_len}, Valid data size: {valid_len}, Test data size: {test_len}")

for e in range(args.num_epochs):
    t0 = time.time()
    e = epoch + e

    # ----------------------------------------------TRAIN---------------------------------------------------
    model.train()
    train_sum, train_weights = 0., 0.
    train_acc = 0.

    total_steps_train = len(loader_train)
    progress_bar_train = tqdm(total=total_steps_train, desc=f"Training Epoch {e+1}", unit="batch", leave=False)
    for i, batch in enumerate(loader_train):

        start_batch = time.time()
        # Feature extraction
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
        X_p, S_p, mask_p, lengths_p, chain_M_p, residue_idx_p, mask_self_p, chain_encoding_all_p = template_featurize(batch, device)
        # X[aa][atom][xyz]
        # S[aa] = 0~20 one-hot encoding
        # chain_encoding_all = [1,1,1,2,2] The first three belong to chain 1, the last two belong to chain 2
        # residue_idx = [0,1,2,3,...,298] Residue index
        # Lengths are the same, and each element corresponds to the others
        elapsed_featurize = time.time() - start_batch
        # mask_for_loss = mask * chain_M
        mask_for_loss = mask
        align_pairs = [[b['pairs'][k]['Align_idx'] for k in b['pairs']] for b in batch]
        tm_scores = [[b['pairs'][k]['TM_score'] for k in b['pairs']] for b in batch]

        optimizer.zero_grad()
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tm_scores)
                # log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

            scaler.scale(loss_av_smoothed).backward()

            if args.gradient_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tm_scores)
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
            loss_av_smoothed.backward()

            if args.gradient_norm > 0.0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

            optimizer.step()
        
        loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)  # true_false: [B,L]
        train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        total_step += 1
        progress_bar_train.set_postfix(Perp=f"{np.exp(train_sum/train_weights):.6f}")
        progress_bar_train.update(1)  # Update progress bar

    progress_bar_train.close()
    train_loss = train_sum / train_weights
    train_accuracy = train_acc / train_weights
    train_perplexity = np.exp(train_loss)
    train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=4)
    train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=4)
    checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'.format(e + 1, total_step)
    torch.save({
        'epoch': e + 1,
        'step': total_step,
        'num_edges': args.num_neighbors,
        'noise_level': args.backbone_noise,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
    }, checkpoint_filename_last)

    if (e + 1) % args.save_model_every_n_epochs == 0:
        checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step)
        torch.save({
            'epoch': e + 1,
            'step': total_step,
            'num_edges': args.num_neighbors,
            'noise_level': args.backbone_noise,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
        }, checkpoint_filename)
    # ------------------------------------------------------------------------------------------------------

    model.eval()
    total_steps_valid = len(loader_valid)
    progress_bar_valid = tqdm(total=total_steps_valid, desc=f"Validating Epoch {e+1}", unit="batch", leave=False)
    validation_sum, validation_weights = 0., 0.
    validation_acc = 0.

    with torch.no_grad():
        for _, batch in enumerate(loader_valid):
            align_pairs = [ [b['pairs'][k]['Align_idx'] for k in b['pairs']] for b in batch ]
            rotations = [[b['pairs'][k]['u'] for k in b['pairs']] for b in batch]
            translations = [[b['pairs'][k]['t'] for k in b['pairs']] for b in batch]
            tm_scores = [[b['pairs'][k]['TM_score'] for k in b['pairs']] for b in batch]
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            X_p, S_p, mask_p, lengths_p, chain_M_p, residue_idx_p, mask_self_p, chain_encoding_all_p = template_featurize(batch, device)
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tm_scores)
            mask_for_loss = mask * chain_M
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            progress_bar_valid.update(1)

    progress_bar_valid.close()
    validation_loss = validation_sum / validation_weights
    validation_accuracy = validation_acc / validation_weights
    validation_perplexity = np.exp(validation_loss)
    validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=4)
    validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=4)

    t1 = time.time()
    dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
    with open(logfile, 'a') as f:
        f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
    print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
