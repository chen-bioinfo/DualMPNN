import warnings
import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio import PDB
from collections import Counter, defaultdict
import ast
import json
warnings.filterwarnings("ignore", module="Bio.PDB")
from Bio.PDB.PDBExceptions import PDBConstructionWarning
class Struct_DualDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []
        print(len(pdb_dict_list))
        for i, entry in enumerate(pdb_dict_list):
            # if len(entry['pairs'])>0:
            if len(entry['target']['seq'])<2048:
                self.data.append(entry)

           
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import numpy as np
from collections import defaultdict

class MultiStruct_Loader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        len_list = []
        pair_nums = []
        for i in range(self.size):
            entry = dataset[i]
            total_len = len(entry['target']['seq'])
            p_num = len(entry['pairs'])
            if p_num!=0:
                for pair in entry['pairs'].values():
                    total_len += len(pair['seq'])
            len_list.append(total_len)
            pair_nums.append(p_num)

        self.len_list = len_list
        pair_groups = defaultdict(list)
        for idx, p_num in enumerate(pair_nums):
            pair_groups[p_num].append((idx, len_list[idx]))

        self.clusters = []
        for p_num, group in pair_groups.items():
            sorted_group = sorted(group, key=lambda x: x[1])
            indices = [x[0] for x in sorted_group]
            lengths = [x[1] for x in sorted_group]

            batch = []
            for ix, length in zip(indices, lengths):
                if self._should_add_to_batch(batch, length):
                    batch.append(ix)
                else:
                    self._add_batch(batch)
                    batch = [ix]
            if batch and not self.drop_last:
                self._add_batch(batch)

    def _should_add_to_batch(self, batch, new_length):
        if not batch:
            return True
        avg_length = sum(self.len_list[i] for i in batch) / len(batch)
        predicted_avg = (avg_length * len(batch) + new_length) / (len(batch) + 1)
        return predicted_avg * (len(batch) + 1) <= self.batch_size

    def _add_batch(self, batch):
        if batch:
            self.clusters.append(batch)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.clusters)
        for cluster in self.clusters:
            yield [self.dataset[i] for i in cluster]

def worker_init_fn(worker_id):
    np.random.seed()

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
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
 


def parse_DualPDBs(dual_loader, pair_util_dict, max_length=10000, num_units=1000000):
    # pair_util_dict = 
    # dual_loader[i] = {target:{...}, pairs:{P1:{...},P2:{...}..}}
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    pdb_dict_list = []
    total_steps = min(len(dual_loader),num_units)

    progress_bar = tqdm(total=total_steps, desc="Processing PDBs", unit="protein")
 
    for step,dicts in enumerate(dual_loader):
        
        # dicts = {target:{...}, pairs:{P1:{...},P2:{...}..}}
        dict_t = dicts['target']
        if dict_t.get('label') is None:
            print(dicts)
        target_ID = dict_t['label']

        dicts_p = dicts['pairs']
        single_dict = {}
        target_dict={}
        pairs_dict={}
        target_dict['seq']=''
        # -------------------------------------TARGET DICT---------------------------------
        concat_seq = ''
        mask_list = []
        visible_list = []
        # print(t['idx'])
        # if 'label' not in list(target_dict):
        #     continue
        if len(list(np.unique(dict_t['idx']))) < 352:
            for idx in list(np.unique(dict_t['idx'])):
                letter = chain_alphabet[idx]
                res = np.argwhere(dict_t['idx']==idx)
                res = res[0]

                try:
                    target_dict['seq_chain_'+letter] = "".join([dict_t['seq'][c] for c in res.tolist()])
                    # target_dict['seq_chain_'+letter] = dict_t['seq']
                    concat_seq += target_dict['seq_chain_'+letter]
                    if idx in dict_t['masked']:
                        mask_list.append(letter)
                    else:
                        visible_list.append(letter)
                    target_dict['coords_chain_'+letter] = dict_t['xyz'][res] # [L_chain, 4, 3] .tolist()
                except Exception as e:
                    print(f"res: {res}")
                    raise
            target_dict['name']= dict_t['label']
            target_dict['masked_list']= mask_list
            target_dict['visible_list']= visible_list
            target_dict['num_of_chains'] = len(mask_list) + len(visible_list)
            target_dict['seq'] = concat_seq

        # ----------------------------------------PAIR DICTS-----------------------------------------
        for k in dicts_p:
            tmp={}
            p_dict = dicts_p[k]
            p_ID = p_dict['label']
            concat_seq = ''
            mask_list = []
            visible_list = []
            if len(list(np.unique(p_dict['idx']))) < 352:
                for idx in list(np.unique(p_dict['idx'])):
                    letter = chain_alphabet[idx]
                    res = np.argwhere(p_dict['idx']==idx)
                    res = res[0]
                    tmp['seq_chain_'+letter]= "".join([p_dict['seq'][c] for c in res.tolist()])
                    concat_seq += tmp['seq_chain_'+letter]
                    if idx in p_dict['masked']:
                        mask_list.append(letter)
                    else:
                        visible_list.append(letter)

                    tmp['coords_chain_'+letter]= p_dict['xyz'][res]
                # parse base info
                tmp['name']= p_dict['label']
                tmp['masked_list']= mask_list
                tmp['visible_list']= visible_list
                tmp['num_of_chains'] = len(mask_list) + len(visible_list)
                tmp['seq'] = concat_seq
                # parse pair info
                # pair_util_dict : {Target1:{p1:{TM-score,AlignedLength,t,u,Aligned_indices},p2:{TM-score,AlignedLength,t,u,Aligned_indices},....}.....}
                tmp['TM_score'] = float(pair_util_dict[target_ID][p_ID]['TM-score'])
                tmp['Align_len'] = pair_util_dict[target_ID][p_ID]['AlignedLength']
                tmp['t'] = [float(num) for num in pair_util_dict[target_ID][p_ID]['t'].split(',')]                              # Translation
                tmp['u'] = [[float(num) for num in row.split(',')] for row in pair_util_dict[target_ID][p_ID]['u'].split(';')]  # Rotation
                tmp['Align_idx'] = ast.literal_eval(pair_util_dict[target_ID][p_ID]['Aligned_indices'])     

            pairs_dict[k]=tmp
        single_dict['target']=target_dict
        single_dict['pairs']=pairs_dict

        pdb_dict_list.append(single_dict)
        progress_bar.update(1)
        if len(pdb_dict_list) >= num_units:
            break
    
    progress_bar.close()
    return pdb_dict_list # pdb_dict_list:[single_dict1,single_dict2,...] # single_dict: {target:{...}, pairs:{P1:{...},P2:{...}..}}


import random
from copy import deepcopy

def build_nested_dict(csv_path):
    # pair_util_dict = {Target1:{p1:{TM-score,AlignedLength,t,u,Aligned_indices},p2:{TM-score,AlignedLength,t,u,Aligned_indices},....}.....}
    result = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id1 = row.pop('Protein1')
            id2 = row.pop('Protein2')
            if id1 not in result:
                result[id1] = {}
            result[id1][id2] = row
    return result

def read_pdb_ids(file_path):
    # Protein1,Protein2,TM-score,AlignedLength,RMSD,SeqID,t,u,Aligned_indices
    df = pd.read_csv(file_path)
    pdb_ids = df.iloc[:, 0].tolist()
    return pdb_ids

def split_clusters(pdb_ids, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # Split into train and temp set
    train_pdb_ids, temp_pdb_ids = train_test_split(pdb_ids, train_size=train_ratio + valid_ratio, random_state=42)
    
    # Split the temp set into validation and test sets
    valid_pdb_ids, test_pdb_ids = train_test_split(temp_pdb_ids, test_size=test_ratio / (valid_ratio + test_ratio), random_state=42)
    
    clusters = {
        'train': train_pdb_ids,
        'validation': valid_pdb_ids,
        'test': test_pdb_ids
    }
    
    return clusters

def build_clusters(param, args):
    pdb_ids = read_pdb_ids("pair_util.csv")
    # clusters = split_clusters(pdb_ids, args.train_ratio, args.valid_ratio, args.test_ratio)
    clusters = split_clusters(pdb_ids, 0.7, 0.25, 0.05)
    return clusters


def CATHloader_pdb(ID, pdb_file_path):
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J",
        "UNK": "X"  
    }
    pdb_id = ID
    parser = PDBParser()
    PREFIX = "%s/%s"%(pdb_file_path,pdb_id)
    
    structure = parser.get_structure('ID', PREFIX+'.pdb')

    seq = ''
    xyz_coords = []
    chain_mark = 0
    idx, masked = [], []
    seq_len = 0

    models = list(structure.get_models())
    if not models:
        print(ID)
        return {}  
    last_model = models[-1]
    ch_len = 0
    for chain in last_model:  
        ch_len += 1
        for residue in chain:
            if residue.get_id()[0] == " ":
                seq_len += 1
                atom_xyz_coords = [
                    list(residue['N'].coord) if 'N' in residue else [np.nan]*3,
                    list(residue['CA'].coord) if 'CA' in residue else [np.nan]*3,
                    list(residue['C'].coord) if 'C' in residue else [np.nan]*3,
                    list(residue['O'].coord) if 'O' in residue else [np.nan]*3
                ]
                xyz_coords.append(atom_xyz_coords)
                
                seq_residue = residue.resname
                letter = three_to_one.get(seq_residue, "X")
                seq += letter
                idx.append(chain_mark)
        chain_mark += 1 
    cnts = Counter(idx)
    long_chains = [x for x in set(idx) if cnts.get(x, 0) >= 10]
    masked = [long_chains]

    return {
        'seq': seq,
        'xyz': torch.tensor(xyz_coords),
        'idx': torch.tensor(idx, dtype=torch.int32),
        'masked': torch.tensor(masked, dtype=torch.int32),
        'label': pdb_id
    }

class Dual_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, data_path, pair_path):
        self.IDs = IDs
        self.loader = loader
        self.data_path = data_path
        self.pair_path = pair_path

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        # IDs = pdb_id
        ID1 = self.IDs[index][0]
        ID2s = self.IDs[index][1]

        target_pdb = self.loader(ID1, self.data_path)
        pair_pdbs = {}
        out = {}
        if ID2s is not None:
            for ID2 in ID2s:
                pair_pdbs[ID2] = self.loader(ID2, self.pair_path)
        # out2 = self.loader(ID2, self.params)
        # out = {target:{...}, pairs:{P1:{...},P2:{...}..}}
        out['target'] = target_pdb
        out['pairs'] = pair_pdbs
        return out  # out = {target:{...}, pairs:{P1:{...},P2:{...}..}}
