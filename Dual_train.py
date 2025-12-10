import argparse
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
from tqdm import tqdm
from model.utils import Struct_DualDataset, MultiStruct_Loader
from model.DualMPNN_util import template_featurize, loss_smoothed, loss_nll, get_std_opt, featurize
from model.DualMPNN_model import DualMPNN


def get_args():
    parser = argparse.ArgumentParser(description='DualMPNN Training Script')
    
    # Path configure
    parser.add_argument("--path_for_outputs", type=str, default="train_outputs/train", help="Output directory")
    parser.add_argument("--train_dataset_path", type=str, default="dataset/CATH/CATH_train.pt", help="Path to training dataset .pt file. You can generate it by template/findTemplate.py")
    parser.add_argument("--valid_dataset_path", type=str, default="dataset/CATH/CATH_test.pt", help="Path to validation dataset .pt file")
  
    # Checkpoint and weight
    parser.add_argument("--previous_checkpoint", type=str, default="", help="Path to checkpoint to load (e.g., output/model_weights/epoch10.pt)")
    parser.add_argument("--reload_model_only", action="store_true", help="If True, only load model weights (for fine-tuning). If False, load optimizer/epoch too (for resuming).")
    
    # Training setup
    parser.add_argument("--device_set", type=str, default="cuda:7", help="Device to use (cuda:0, cpu)")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--save_model_every_n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8000, help="Number of tokens per batch")
    parser.add_argument("--max_protein_length", type=int, default=200000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone_noise", type=float, default=0.02)
    parser.add_argument("--gradient_norm", type=float, default=3.0)
    parser.add_argument("--mixed_precision", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device(args.device_set if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    logfile = base_folder + 'log.txt'
    if not args.previous_checkpoint or args.reload_model_only:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
    if args.debug:
        args.num_epochs = 2
        args.batch_size = 2000
        print("Debug mode enabled: reduced epochs and batch size.")

    print(f"Loading training data from: {args.train_dataset_path}")
    pdb_dict_train = torch.load(args.train_dataset_path)
    dataset_train = Struct_DualDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
    loader_train = MultiStruct_Loader(dataset_train, batch_size=args.batch_size)

    print(f"Loading validation data from: {args.valid_dataset_path}")
    pdb_dict_valid = torch.load(args.valid_dataset_path)
    dataset_valid = Struct_DualDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length) 
    loader_valid = MultiStruct_Loader(dataset_valid, batch_size=args.batch_size)
    model = DualMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)
    total_step = 0
    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    start_epoch = 0
    best_validation_accuracy = -float("inf")
    best_checkpoint_path = base_folder + "model_weights/best_model.pt"

    if args.previous_checkpoint:
        if os.path.isfile(args.previous_checkpoint):
            print(f"Loading checkpoint '{args.previous_checkpoint}'")
            checkpoint = torch.load(args.previous_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            if args.reload_model_only:
                print("Mode: Fine-tuning / Transfer Learning")
                print("Loaded model weights only. Optimizer, Epoch, and Step count reset.")
            else:
                print("Mode: Resuming Training")
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                total_step = checkpoint['step']
                if 'best_validation_accuracy' in checkpoint:
                    best_validation_accuracy = checkpoint['best_validation_accuracy']
                
                print(f"Resumed from Epoch {start_epoch}, Step {total_step}")
        else:
            print(f"No checkpoint found at '{args.previous_checkpoint}'. Starting from scratch.")

    train_len = len(dataset_train)
    valid_len = len(dataset_valid)
    print(f"Train data size (proteins): {train_len}, Valid data size (proteins): {valid_len}")

    for e in range(start_epoch, args.num_epochs):
        t0 = time.time()
        
        # ----------------------------------------------TRAIN---------------------------------------------------
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.

        total_steps_train = len(loader_train)
        progress_bar_train = tqdm(total=total_steps_train, desc=f"Training Epoch {e+1}", unit="batch", leave=False)
        
        for i, batch in enumerate(loader_train):
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            X_p, S_p, mask_p, lengths_p, chain_M_p, residue_idx_p, mask_self_p, chain_encoding_all_p = template_featurize(batch, device)

            mask_for_loss = mask
            align_pairs = [[b['pairs'][k]['Align_idx'] for k in b['pairs']] for b in batch]
            rotations = [[b['pairs'][k]['u'] for k in b['pairs']] for b in batch]
            translations = [[b['pairs'][k]['t'] for k in b['pairs']] for b in batch]
            tm_scores = [[b['pairs'][k]['TM_score'] for k in b['pairs']] for b in batch]

            optimizer.zero_grad()
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tm_scores)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                scaler.scale(loss_av_smoothed).backward()
                
                if args.gradient_norm > 0.0:
                    scaler.unscale_(optimizer.optimizer)
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
            
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss) # true_false: [B,L]
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            total_step += 1
            progress_bar_train.set_postfix(Perp=f"{np.exp(train_sum/train_weights):.4f}")
            progress_bar_train.update(1) 

        progress_bar_train.close()
        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=4)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=4)

        # Last Checkpoint
        checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'
        torch.save({
            'epoch': e + 1,
            'step': total_step,
            'num_edges': args.num_neighbors,
            'noise_level': args.backbone_noise,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'best_validation_accuracy': best_validation_accuracy
        }, checkpoint_filename_last)

        # Every 5 epoch checkpoint
        if (e + 1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step)
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'best_validation_accuracy': best_validation_accuracy
            }, checkpoint_filename)

        # ----------------------------------------------VALIDATION----------------------------------------------
        model.eval()
        total_steps_valid = len(loader_valid)
        progress_bar_valid = tqdm(total=total_steps_valid, desc=f"Validating Epoch {e+1}", unit="batch", leave=False)
        validation_sum, validation_weights = 0., 0.
        validation_acc = 0.
        
        with torch.no_grad():
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                X_p, S_p, mask_p, lengths_p, chain_M_p, residue_idx_p, mask_self_p, chain_encoding_all_p = template_featurize(batch, device)
                
                align_pairs = [ [b['pairs'][k]['Align_idx'] for k in b['pairs']] for b in batch ]
                rotations = [[b['pairs'][k]['u'] for k in b['pairs']] for b in batch]
                translations = [[b['pairs'][k]['t'] for k in b['pairs']] for b in batch]
                tm_scores = [[b['pairs'][k]['TM_score'] for k in b['pairs']] for b in batch]

                log_probs= model(X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p, align_pairs, tm_scores)
                
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

        # Best Model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = e + 1
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'best_validation_accuracy': best_validation_accuracy,
            }, best_checkpoint_path)
            print(f"New best model saved (epoch {best_epoch}, valid_acc: {validation_accuracy_})")
            with open(logfile, 'a') as f:
                f.write(f'New best model: epoch {best_epoch}, best_acc: {validation_accuracy_}\n')

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            
        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')

if __name__ == "__main__":
    main()
