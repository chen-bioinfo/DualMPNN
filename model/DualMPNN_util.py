from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import native_group_norm, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


def initialize_h_V(self, E, S, W_s, onehot_dim=21, init_ratio=0.0, error_init_ratio=0.0):
    """
    Initialize `h_V` where 20% of amino acid nodes are represented by their corresponding one-hot encoding,
    and the rest remain zero.

    Args:
        E: Edge feature tensor [B, L, edge_features].
        S: Sequence tensor [B, L], integers from 0-20 representing amino acid types.
        W_s: Embedding layer to project amino acids into hidden dimensions.
        onehot_dim: Dimension of one-hot encoding (default is 21 for amino acids).
        init_ratio: Percentage of nodes initialized with one-hot (default is 20%).
        error_init_ratio: Percentage of nodes with incorrect initialization (default is 10%).

    Returns:
        h_V: Initialized node feature tensor [B, L, edge_features].
    """
    device = E.device
    B, L = E.shape[:2]

    # Initialize `h_V` as all zeros
    h_V = torch.zeros((B, L, E.shape[-1]), device=device)

    # Generate a random mask for nodes to initialize with one-hot encoding
    total_elements = B * L
    num_init_elements = int(total_elements * init_ratio)
    all_indices = torch.arange(total_elements, device=device)
    init_indices = all_indices[torch.randperm(total_elements)[:num_init_elements]]
    random_mask = torch.zeros(total_elements, device=device, dtype=torch.bool)
    random_mask[init_indices] = True
    random_mask = random_mask.view(B, L)

    # Map sequence tensor `S` to embeddings using `W_s`
    embedded_S = W_s(S)  # [B, L, hidden_dim]
    h_V[random_mask] = embedded_S[random_mask]

    # Create a separate mask for incorrect initialization
    num_error_elements = int(total_elements * error_init_ratio)
    available_indices = all_indices[~random_mask.view(-1)]
    error_indices = available_indices[torch.randperm(len(available_indices))[:num_error_elements]]
    error_mask = torch.zeros(total_elements, device=device, dtype=torch.bool)
    error_mask[error_indices] = True
    error_mask = error_mask.view(B, L)

    # Generate incorrect one-hot representations, ensuring no conflicts with original `S`
    random_S = torch.randint(0, onehot_dim, (B, L), device=device)
    conflict_mask = (random_S == S) & error_mask
    while conflict_mask.any():
        random_S2 = torch.randint(0, onehot_dim, (B, L), device=device)
        random_S[conflict_mask] = random_S2[conflict_mask]
        conflict_mask = (random_S == S) & error_mask

    error_embedded_S = W_s(random_S)
    h_V[error_mask] = error_embedded_S[error_mask]

    return h_V
def template_featurize(batch, device):
    """
    Featurize batch data for template pairs.

    Args:
        batch: Input batch data containing pairs.
        device: Target PyTorch device (e.g., "cuda" or "cpu").

    Returns:
        Tuple of tensors split across the second dimension (K templates for each protein).
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    batch = [b['pairs'] for b in batch]
    B = len(batch)
    if B == 0:
        return tuple()
    
    # Determine the number of sub-dictionaries (K) for each sample
    K = len(batch[0])

    # Precompute the maximum sequence length among all sub-dictionaries
    all_sub_lengths = []
    for b in batch:
        for sub_b in b.values():
            chains = sub_b['masked_list'] + sub_b['visible_list']
            total_length = sum(len(sub_b[f'seq_chain_{letter}']) for letter in chains)
            all_sub_lengths.append(total_length)
    L_max = max(all_sub_lengths) if all_sub_lengths else 0

    # Initialize arrays with new dimensions
    X = np.full([B, K, L_max, 4, 3], np.nan)
    residue_idx = -100 * np.ones([B, K, L_max], dtype=np.int32)
    chain_M = np.zeros([B, K, L_max], dtype=np.int32)
    mask_self = np.ones([B, K, L_max, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, K, L_max], dtype=np.int32)
    S = np.zeros([B, K, L_max], dtype=np.int32)
    lengths = np.zeros([B, K], dtype=np.int32)

    # Alphabet initialization for chain encoding
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    extra_alphabet = [str(item) for item in range(300)]
    chain_letters = init_alphabet + extra_alphabet

    for i, b in enumerate(batch):
        for k, (sub_key, sub_b) in enumerate(b.items()):
            # Initialize variables for processing chains
            masked_chains = sub_b['masked_list'].copy()
            visible_chains = sub_b['visible_list'].copy()
            all_chains = masked_chains + visible_chains
            random.shuffle(all_chains)

            x_chain_list = []
            chain_mask_list = []
            chain_seq_list = []
            chain_encoding_list = []
            c = 1
            l0, l1 = 0, 0

            for letter in all_chains:
                chain_seq = sub_b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)

                chain_mask = np.ones(chain_length) if letter in masked_chains else np.zeros(chain_length)
                x_chain_list.append(sub_b[f'coords_chain_{letter}'])
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding = c * np.ones(chain_length)
                chain_encoding_list.append(chain_encoding)

                l1 += chain_length
                mask_self[i, k, l0:l1, l0:l1] = 0
                residue_idx[i, k, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 = l1
                c += 1

            # Merge chain data
            try:
                x = np.concatenate(x_chain_list, axis=0)
                all_sequence = ''.join(chain_seq_list)
                l = len(all_sequence)
                lengths[i, k] = l

                # Pad to the maximum sequence length
                X[i, k] = np.pad(x, [(0, L_max - l)] + [(0, 0)] * 2, constant_values=np.nan)
                chain_M[i, k] = np.pad(np.concatenate(chain_mask_list), (0, L_max - l))
                chain_encoding_all[i, k] = np.pad(np.concatenate(chain_encoding_list), (0, L_max - l))
                S[i, k, :l] = [alphabet.index(a) for a in all_sequence]
            except Exception as e:
                print(f"Error processing {sub_key}: {e}")
                raise

    # Post-processing
    mask = np.isfinite(np.sum(X, axis=(-2, -1))).astype(np.float32)
    X[np.isnan(X)] = 0

    # Convert to PyTorch tensors
    tensor_args = {
        'dtype': torch.long,
        'device': device
    }

    X = torch.from_numpy(X).float().to(device)
    S = torch.from_numpy(S).to(**tensor_args)
    mask = torch.from_numpy(mask).float().to(device)
    lengths = torch.from_numpy(lengths).to(**tensor_args)
    chain_M = torch.from_numpy(chain_M).float().to(device)
    residue_idx = torch.from_numpy(residue_idx).to(**tensor_args)
    mask_self = torch.from_numpy(mask_self).float().to(device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(**tensor_args)

    def split_tensor(tensor):
        """Split a tensor along the second dimension."""
        return torch.unbind(tensor, dim=1)

    X_split = split_tensor(X)
    S_split = split_tensor(S)
    mask_split = split_tensor(mask)
    lengths_split = split_tensor(lengths)
    chain_M_split = split_tensor(chain_M)
    residue_idx_split = split_tensor(residue_idx)
    mask_self_split = split_tensor(mask_self)
    chain_encoding_all_split = split_tensor(chain_encoding_all)

    return (
        X_split, S_split, mask_split, lengths_split,
        chain_M_split, residue_idx_split, mask_self_split, chain_encoding_all_split
    )

def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['target']['seq']) for b in batch], dtype=np.int32) # Sum of chain sequence lengths

    L_max = max([len(b['target']['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) # Residue indices with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) # 1.0 for positions to predict, 0.0 for given positions
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) # For interface loss calculation - 0.0 for self-interactions, 1.0 for others
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) # Integer encoding for chains: 0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2...
    S = np.zeros([B, L_max], dtype=np.int32) # Sequence as integers (AAs)
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        b = b['target']
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) # Randomly shuffle chain order
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_mask = np.zeros(chain_length) # 0.0 for visible chains
                x_chain_list.append(b[f'coords_chain_{letter}'])        
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_mask = np.ones(chain_length) # 1.0 for masked chains
                x_chain_list.append(b[f'coords_chain_{letter}'])    
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        # try:
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for positions to predict
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        # except Exception as e:
        #     print(b)
        #     raise
        

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16, CA_only=False):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.CA_only = CA_only
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        if not self.CA_only:
            node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        else:
            node_in, edge_in = 6, num_positional_embeddings + num_rbf*1
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Ca = X[:,:,1,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        if not self.CA_only:
            Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
            N = X[:,:,0,:]
            C = X[:,:,2,:]
            O = X[:,:,3,:]
            RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
            RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
            RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
            RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
            RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.5, 0.98), eps=1e-9), step
    )


