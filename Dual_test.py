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
from model.utils import  worker_init_fn,parse_DualPDBs, build_nested_dict, Dual_dataset,CATHloader_pdb
from model.DualMPNN_util import template_featurize, loss_nll, get_std_opt,featurize
from model.DualMPNN_model import DualMPNN
from multiprocessing import Value, Lock
from tqdm import tqdm

def get_seq(S):
    sequences=[]
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    for sample in S.cpu().numpy():  # Iterate over each sample
        seq = ''.join([alphabet[i] for i in sample])
        sequences.append(seq)
    return sequences
def save_to_txt(rec_rate, rec_perp, output_path, tar_id, pair_id, S, S_pred, tm):
    S = get_seq(S)[0]
    S_pred = get_seq(S_pred)[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = output_path + f"/{tar_id}.txt"
    with open(output_path, 'w') as f:
        f.write(f"Target: {tar_id}\n")
        f.write(f"Origin seq:  {S}\n")
        f.write(f"Sampled seq: {S_pred}\n")
        f.write(f"Recovery: {rec_rate:.4f}\n")
        f.write(f"Perplexity: {rec_perp:.4f}\n")
        f.write(f"Utilizing Pairs: {pair_id}\n")
        f.write(f"TM Score: {tm}\n")

class MyArgs(object):
  def __init__(self):
    #--------------------------------------------------------#
    self.device_set = "cuda:0"
    self.path_testset = "dataset/T500/Test500.pt"
    self.path_for_outputs = ""                                  # Path for sequence generated
    self.model_path = ""                                        # Path for train model checkpoint
    #--------------------------------------------------------#
    self.num_epochs = 200
    self.save_model_every_n_epochs = 5
    self.reload_data_every_n_epochs = 4
    self.num_examples_per_epoch = 200000
    self.num_examples_valid = 20000
    self.batch_size = 10
    self.max_protein_length = 2000
    self.hidden_dim = 128
    self.num_encoder_layers = 4
    self.num_decoder_layers = 4
    self.num_neighbors = 48
    self.dropout = 0.1
    self.backbone_noise = 0.0
    self.rescut = 3.5
    self.debug = False
    self.gradient_norm = -1.0 
    self.mixed_precision= True

args = MyArgs()

scaler = torch.cuda.amp.GradScaler()

base_folder = time.strftime(args.path_for_outputs, time.localtime())

if base_folder[-1] != '/':
    base_folder += '/'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

PATH = args.model_path
dataset_test = torch.load(args.path_testset)

model = DualMPNN(node_features=args.hidden_dim, 
                    edge_features=args.hidden_dim, 
                    hidden_dim=args.hidden_dim, 
                    num_encoder_layers=args.num_encoder_layers, 
                    num_decoder_layers=args.num_encoder_layers, 
                    k_neighbors=args.num_neighbors, 
                    dropout=args.dropout, 
                    augment_eps=args.backbone_noise)


device = torch.device(args.device_set if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    test_acc = 0.
    total_steps_test = len(dataset_test)
    progress_bar_test = tqdm(total=total_steps_test, unit="pdb", leave=False)
    all_probs_list = []
    all_log_probs_list = []
    S_sample_list = []
    recover_len = 0
    total_len = 0
    for _, batch in enumerate(dataset_test):
        batch = [batch]
        align_pairs = [ [b['pairs'][k]['Align_idx'] for k in b['pairs']] for b in batch ]
        tm_scores = [[b['pairs'][k]['TM_score'] for k in b['pairs']] for b in batch]
        
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
        X_p, S_p, mask_p, lengths_p, chain_M_p, residue_idx_p, mask_self_p, chain_encoding_all_p = template_featurize(batch, device)

        accumulated_probs = torch.zeros((X.shape[0], X.shape[1], 21), device=device)
        
        num_samples = 1
        all_samples = []
        all_probs = []
        accumulated_probs = torch.zeros((X.shape[0], X.shape[1], 21), device=device)
        for run in range(num_samples):
            seed = run
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            S_sample = model.sample(X, mask, chain_M, residue_idx, chain_encoding_all, 
                                    X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p,
                                    align_pairs, tm_scores, temperature=0.01)
            
            log_probs = model(X, S_sample, mask, chain_M, residue_idx, chain_encoding_all,
                        X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p,
                        align_pairs, tm_scores)
            all_samples.append(S_sample)
            accumulated_probs += torch.exp(log_probs)
        average_probs = accumulated_probs / num_samples
        samples_tensor = torch.stack(all_samples, dim=0) # (num_samples, batch, seq_len)
        one_hot = F.one_hot(samples_tensor, num_classes=21) 
        counts = torch.sum(one_hot, dim=0)
        final_pred = torch.argmax(counts, dim=-1)
        num_pairs = len(X_p)
        mask_for_loss = mask * chain_M
        seq_recovery_rate = torch.sum(
            torch.sum(torch.nn.functional.one_hot(S[0], 21) * 
                     torch.nn.functional.one_hot(final_pred[0], 21), dim=-1) * mask_for_loss[0]
        ) / torch.sum(mask_for_loss[0])
        log_average_probs = torch.log(average_probs + 1e-10)
        loss, loss_av, true_false = loss_nll(S, log_average_probs, mask_for_loss)
        S_argmaxed = torch.argmax(log_average_probs,-1)
        test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
        test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
        test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        denominator = torch.sum(mask_for_loss).cpu().data.numpy()
        i_acc = torch.sum(true_false * mask_for_loss).cpu().data.numpy() / (denominator + 1e-10)
        i_perp = np.exp(torch.sum(loss * mask_for_loss).cpu().data.numpy() / (denominator + 1e-10))

        if num_pairs>0:
            ids = []
            for k in batch[0]['pairs']:
                ids.append(k)
            save_to_txt(i_acc, i_perp, base_folder, batch[0]['target']['name'], ids, S, S_argmaxed, tm_scores[0])
        else:
            save_to_txt(i_acc, i_perp, base_folder, batch[0]['target']['name'], 'no pairs', S, S_argmaxed, tm_scores[0])
        seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), 
                                                   unique=False, precision=4)
        
        total_len += len(S[0])
        recover_len += len(S[0]) * float(seq_rec_print)

        progress_bar_test.set_postfix(rec_acc=f"{(test_acc/test_weights):.4f}")
        progress_bar_test.update(1)
        
progress_bar_test.close()
test_loss = test_sum / test_weights
test_rec = test_acc/test_weights
test_perplexity = np.exp(test_loss)
test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=4)
test_accuracy_ = np.format_float_positional(np.float32(test_acc), unique=False, precision=4)
print(f'checkpoint: {os.path.basename(args.model_path)}, test_perp: {test_perplexity_}, test_rec:{test_rec}')
