import os
import pandas as pd
from multiprocessing import Pool, Lock
from collections import defaultdict
import json
from tqdm import tqdm
import csv
import subprocess
import numpy as np
from Bio.PDB import PDBParser, PPBuilder, Superimposer
import re
import torch
from collections import Counter
import ast, time, argparse
import warnings,shutil
warnings.filterwarnings("ignore", module="Bio.PDB")

def get_missing_files(output_path,processed_files):
    missing = []
    for pdb_file in processed_files:
        parent_dir = os.path.dirname(pdb_file)
        rel_path = os.path.basename(parent_dir)
        output_dir=os.path.join(output_path,rel_path)
        root, file = os.path.split(pdb_file)
        filename = os.path.splitext(file)[0]
        aln_path = os.path.join(output_dir, f"{filename}.txt")
        if not os.path.exists(aln_path):
            missing.append(pdb_file)
    return missing


def process_file(foldseek_path,foldseek_database_path,output_path,temp_path,pdb_file):
    try:
        parent_dir = os.path.dirname(pdb_file)
        rel_path = os.path.basename(parent_dir)
        output_dir=os.path.join(output_path,rel_path)
        os.makedirs(output_dir, exist_ok=True)
        root, file = os.path.split(pdb_file)
        filename, ext = os.path.splitext(file)
        input_path = pdb_file
        oup = os.path.join(output_dir, filename + '.txt')
        unique_tmp = os.path.join(temp_path, f"foldseek_tmp_{filename}")
        os.makedirs(unique_tmp, exist_ok=True)
        # cmd = foldseek_path + ' easy-search ' + input_path+ ' ' + foldseek_database_path+ ' ' + oup +' '+ temp_path + ' --format-output query,target,qtmscore,alnlen,u,t  --threads 1'
        cmd = [
            foldseek_path, "easy-search",
            input_path,
            foldseek_database_path,
            oup,
            unique_tmp,
            "--format-output", "query,target,qtmscore,alnlen,u,t",
            "--threads", "1"
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"[ERROR] Foldseek failed for {filename}:")
                print(result.stderr.strip())
                return False

        except Exception as e:
            print(f"[EXCEPTION] {filename}: {e}")
            return False
        finally:
        
            shutil.rmtree(unique_tmp, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        return False

def parse_folder(root_dir):
    final_dict = defaultdict(dict)
    
    for root, dirs, files in os.walk(root_dir):
        for filename in tqdm(files, desc=f"Files in {os.path.basename(root)}", leave=False):
            if not filename.lower().endswith('.txt'):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, root_dir)
            file_dict = parse_foldseek_results(file_path)

            for key, chains in file_dict.items():
                if len(key)>6:
                    key = key[:6]
                final_dict[key] = {
                    'chains': chains,
                    'cluster': rel_path if rel_path != '.' else ''
                }
    
    return dict(final_dict)
    
def parse_foldseek_results(file_path):

    result = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            if '_' in cols[0]:
                left_id = cols[0].split('_')[0] + '_' + cols[0].split('_')[-1]
            else:
                left_id = cols[0]
            right_chain = cols[1].split('.')[0] + '_' + cols[1].split('_')[-1]
            similarity = float(cols[2])
            if similarity >= 0.990 or left_id[:4] == right_chain[:4] or left_id[:3]==right_chain[:3]:
                continue
            result[left_id].append((similarity, right_chain))

    for k in result:
        result[k] = [chain for _, chain in sorted(result[k], key=lambda x: (-x[0], x[1]))][:2]
    return dict(result)

def process_file_wrapper(args):
    return process_file(*args)

def find_pairs(target_path,output_path,foldseek_path,temp_path,foldseek_database_path,n_proc):
    paths = [output_path, temp_path]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create directory: {path}")
    pdb_files = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))

    lock = Lock()
    with lock:
        os.makedirs(output_path, exist_ok=True)

    n_processes = n_proc

    with Pool(processes=n_processes) as pool:
        args = [
            (foldseek_path, foldseek_database_path, output_path, temp_path, pdb_file)
            for pdb_file in pdb_files
        ]
        # results = pool.starmap(process_file, args)
        # results = pool.map(process_file,foldseek_path,foldseek_database_path,output_path,temp_path, pdb_files)
        results = []
        with Pool(processes=n_proc) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_file_wrapper, args),
                total=len(args),
                desc="Running Foldseek",
                dynamic_ncols=True
            ))
    success = sum(results)
    print(f"[INFO] Completed: {success}/{len(args)} successful searches.")

    failed = sum(1 for r in results if not r)
    print(f"Processed {len(results)-failed} files, {failed} failed")

    for retry in range(3):
        n_processes = int(n_processes/4) + 1
        missing_files = get_missing_files(output_path,pdb_files)
        print(len(missing_files))
        if not missing_files:
            break
        with Pool(processes=n_processes) as pool:
            args = [
                (foldseek_path, foldseek_database_path, output_path, temp_path, pdb_file)
                for pdb_file in missing_files
            ]
            results = pool.starmap(process_file, args)
            # results = pool.map(process_file,foldseek_path,foldseek_database_path,output_path,temp_path, pdb_files)
        for pdb_file in missing_files:
            parent_dir = os.path.dirname(pdb_file)
            rel_path = os.path.basename(parent_dir)
            output_dir=os.path.join(output_path,rel_path)
            os.makedirs(output_dir, exist_ok=True)
            root, file = os.path.split(pdb_file)
            filename, ext = os.path.splitext(file)
            input_path = pdb_file
            oup = output_dir+ '/' + filename + '.txt'

            unique_tmp = os.path.join(temp_path, f"foldseek_tmp_{filename}")
            os.makedirs(unique_tmp, exist_ok=True)
            # cmd = foldseek_path + ' easy-search ' + input_path+ ' ' + foldseek_database_path+ ' ' + oup +' '+ unique_tmp + ' --format-output query,target,qtmscore,alnlen,u,t  --threads 1'   
            # os.system(cmd)
            cmd = [
                foldseek_path, "easy-search",
                input_path,
                foldseek_database_path,
                oup,
                unique_tmp,
                "--format-output", "query,target,qtmscore,alnlen,u,t",
                "--threads", "1"
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            shutil.rmtree(unique_tmp, ignore_errors=True)
        pdb_files = missing_files
    final_missing = get_missing_files(output_path,pdb_files)
    if final_missing:
        print(f"\nWarning: Still {len(final_missing)} files are missing. Please check manually.")
    else:
        print("\nAll files have been successfully generated.")


def sanitize_chid(chid):
    if not chid: 
        return "UNK"
    return re.sub(r'[^A-Za-z0-9]', '_', chid)[:3]

def split_pdb_by_chain(input_path, output_dir):
    pdbid = os.path.splitext(os.path.basename(input_path))[0]

    os.makedirs(output_dir, exist_ok=True)
    
    handles = {}
    
    try:
        with open(input_path, 'r') as f_in:
            for line_num, line in enumerate(f_in, 1):
                if line.startswith(('ATOM', 'HETATM', 'TER')):
                    raw_chid = line[21] if len(line) > 21 else ' '
                    chid = raw_chid.strip() or ' '
                    
                    sanitized = sanitize_chid(chid)
                    
                    if sanitized not in handles:
                        output_path = os.path.join(
                            output_dir, 
                            f"{pdbid}_{sanitized}.pdb"
                        )
                        handles[sanitized] = open(output_path, 'w')
                    
                    handles[sanitized].write(line)
                
                elif line.startswith('HEADER'):
                    for handle in handles.values():
                        handle.write(line)
    
    except Exception as e:
        print(f"Error processing file: {str(e)} (line {line_num})")
        raise
    
    finally:
        for handle in handles.values():
            handle.close()
    
    return [os.path.abspath(f.name) for f in handles.values()]

def run_tmalign(pdb1, pdb2, matrix=None):
    try:
        if len(matrix) == 0:
            cmd = f"TM_Align/TMalign {pdb1} {pdb2}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        else:
            cmd = f"TM_Align/TMalign {pdb1} {pdb2} -m {matrix}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return result.stdout
    except Exception as e:
        print(f"Running TMalign Error: {e}")
    return None


def parse_indices(output):
    aligned_indices = []
    lines = output.splitlines()
    
    seq_line_1 = ""
    seq_line_2 = ""
    seq_line_3 = ""
    
    for i, line in enumerate(lines):
        if line.startswith("(You should use TM-score"):  
            seq_line_1 = lines[i + 3]  
            seq_line_2 = lines[i + 4]  
            seq_line_3 = lines[i + 5]  

    cnt1=0
    cnt2=0
    for i, (aa1, aa2, aa3) in enumerate(zip(seq_line_1, seq_line_2, seq_line_3)):
        if aa1 == '-':
            cnt1+=1
        if aa3 == '-':
            cnt2+=1
        if aa2 == ':':
            aligned_indices.append([i-cnt1, i-cnt2])

    return aligned_indices

def parse_tmalign(output):
    tm_score = 0
    aligned_length = 0
    rmsd = -1
    seq_id = None

    try:
        lines = output.splitlines()
        for line in lines:
            if line.startswith("Aligned length="):
                parts = line.split(",")
                if len(parts) >= 3:
                    aligned_length = int(parts[0].split("=")[1].strip())
                    rmsd = float(parts[1].split("=")[1].strip())
                    seq_id_str = parts[2].split("=")[-1].strip()  
                    seq_id = float(seq_id_str)
            if line.startswith("TM-score="):
                if tm_score == 0:  
                    tm_score = float(line.split()[1])

    except Exception as e:
        print(f"Parsing TMalign Output Error: {e}")

    return tm_score, aligned_length, rmsd, seq_id
def parse_TR(matrix_file):
    rotation_matrix = None
    translation_vector = None
    with open(matrix_file, "r") as f:
        matrix_lines = f.readlines()

        matrix_lines = matrix_lines[2:5]

        rotation_matrix = np.array([list(map(float, line.split()[2:])) for line in matrix_lines])
        rotation_matrix = rotation_matrix.T

        translation_vector = np.array([float(line.split()[1]) for line in matrix_lines])

    return rotation_matrix, translation_vector
def process_pair(pdb1, pdb_files, output_dir, pair_info_k):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'/mat'):
        os.makedirs(output_dir+'/mat', exist_ok=True)
    _ , filename = os.path.split(pdb1)
    pdb1_id, ext = os.path.splitext(filename)

    data_list = []
    for i,pdb2 in enumerate(pdb_files):
        pdb2_id = os.path.splitext(os.path.basename(pdb2))[0]
        matrix_file = os.path.join(output_dir+'/mat', f'mat_{pdb1_id}_{pdb2_id}')
        output = run_tmalign(pdb1, pdb2, matrix_file)
        result = parse_tmalign(output)
        align_inx = parse_indices(output)
        if result == (0,0,-1,None):
            print(f"Run TM-align error. Please check if the pdb file {pdb1} or {pdb2} is valid.")
        else:
            tm_score, aligned_length, rmsd, seq_id = result
            rotation_matrix = None
            translation_vector = None
            rotation_matrix, translation_vector = parse_TR(matrix_file)
            rotation_matrix = rotation_matrix if rotation_matrix is not None else "N/A"
            translation_vector = translation_vector if translation_vector is not None else "N/A" 
            rotation_matrix_str = ';'.join([','.join(map(str, row)) for row in rotation_matrix])
            translation_vector_str = ','.join(map(str, translation_vector))
            list_pdb = [pdb1_id, pdb2_id, tm_score, aligned_length, rmsd, seq_id, translation_vector_str, rotation_matrix_str, align_inx, pdb1, pdb2]
            data_list.append(list_pdb)
    return data_list
def process_single_entry(args):
    k, pair_info_k, input_path, input_pairs_path, output_path = args
    data_list = []

    pdb1 = os.path.join(input_path, f"{k}.pdb")
    # ssplit_pdb_by_chain(pdb1, os.path.join(output_path,'chains'))
    # pdb2_list = [os.path.join(input_pairs_path, f"{pair_info_k[item][0].split('.')[0]}_{pair_info_k[item][0].split('_')[-1]}.pdb")
    #              for item in pair_info_k]
    pdb2_list = [os.path.join(input_pairs_path, f"{item}.pdb")
                    for item in pair_info_k['chains'] ]
    if not os.path.exists(pdb1):
        return []
    data_list = process_pair(pdb1, pdb2_list, output_path, pair_info_k)
    
    return data_list

def worker_init_fn(worker_id):
    np.random.seed()


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

def run_generate_dataset(target_path, pair_path, file_json, file_csv, file_pt, num_worker, max_protein_length=200000, num_examples_valid=200000):
    test_clu=[]
    for filename in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, filename)):
            name_without_ext = os.path.splitext(filename)[0]
            test_clu.append(name_without_ext)
    with open(file_json, 'r') as f:
        cluster = json.load(f)
    LOAD_PARAM_VALID = {'batch_size': None,
                        'shuffle': True,
                        'pin_memory':False,
                        'num_workers': num_worker,
                        'persistent_workers': True}
    pair_util_dict = build_nested_dict(file_csv)
    test_cluster = [[pdbid, cluster[pdbid]["chains"] if cluster.get(pdbid) is not None else None] for pdbid in test_clu]
    test_set = Dual_dataset(test_cluster, CATHloader_pdb, target_path, pair_path)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM_VALID)
    pdb_dict_test = parse_DualPDBs(test_loader, pair_util_dict, max_protein_length, num_examples_valid)
    torch.save(pdb_dict_test, file_pt)

def run_generate_csv(json_path,input_path,input_pairs_path,output_path, output_csv, n_proc_merge):
    
    with open(json_path, 'r') as f:
        pair_json = json.load(f)
    output_csv = output_csv

    args_list = [(k, pair_json[k], input_path, input_pairs_path, output_path) for k in pair_json]
    print(args_list[0])
    with Pool(processes=n_proc_merge) as pool:
        results = list(tqdm(pool.imap(process_single_entry, args_list),
                            total=len(args_list),
                            desc="Processing Protein Pairs"))

    data = []
    for result in results:
        data.extend(result)

    header = ['Protein1', 'Protein2', 'TM-score', 'AlignedLength', 'RMSD', 
             'SeqID', 't', 'u', 'Aligned_indices','Path1','Path2']
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(data)

def main():
    parser = argparse.ArgumentParser(description=
        "Generate protein pair dataset pipeline. Given a dataset folder, this script will find best protein pairs with Foldseek and generate a dataset .pt that contains all info that model needs.")
    
    parser.add_argument('--dataset_name', type=str, default='CATH_train',
                        help='Name your dataset, e.g., CATH T500 or SCOPe')
    parser.add_argument('--target_path', type=str, default='../training/dataset/CATH/train',
                        help='Path to your dataset folder. If your folder contains train and test subfolders, please specify the train folder. And run this script respectively for train and test folders.')
    parser.add_argument('--foldseek_path', type=str, default='../foldseek/bin/foldseek',
                        help='Path to Foldseek binary')
    parser.add_argument('--foldseek_db', type=str, default='../foldseek/pdb',
                        help='Path to Foldseek database file')
    parser.add_argument('--input_pairs_path', type=str, default='../foldseek/PDBdb/',
                        help='Path to input PDB pairs folder')
    parser.add_argument('--n_proc_find', type=int, default=128, help='Number of processes for finding pairs. The recommended value is your cpu cores.')
    parser.add_argument('--n_proc_merge', type=int, default=128, help='Number of processes for merging results. The recommended value is your cpu cores.')
    parser.add_argument('--temp_path', type=str, default='tmp_3', help='Temporary folder for foldseek tmp files')

    args = parser.parse_args()

    # === path ===
    dataset_name = args.dataset_name
    output_path = f'{dataset_name}_pairs'
    json_name = f'/{dataset_name}_pairs.json'
    csv_name = f'/{dataset_name}.csv'
    outcsv = output_path + csv_name
    outpt = output_path + f'/{dataset_name}.pt'
    output_json = output_path + json_name
    os.makedirs(output_path, exist_ok=True)

    # === Step 1 ===
    print(f"\n===== STEP 1: Finding candidate pairs using Foldseek =====")
    print(f"\nIt takes about 1 hour to process 20000 PDBs by 128 cpu cores.")
    time_start = time.time()
    find_pairs(args.target_path, output_path, args.foldseek_path, args.temp_path, args.foldseek_db, args.n_proc_find)
    print(f'Finish STEP 1.')
    print(f'Cost time for STEP 1: {time.time() - time_start:.2f}s')

    # === Step 2 ===
    print(f"\n===== STEP 2: Parsing and filtering candidates =====")
    time_start = time.time()
    result = parse_folder(output_path)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'Finish STEP 2.')
    print(f'Cost time for STEP 2: {time.time() - time_start:.2f}s')

    # === Step 3 ===
    print(f"\n===== STEP 3: Running TM-align and generating CSV =====")
    time_start = time.time()
    run_generate_csv(output_json, args.target_path, args.input_pairs_path, output_path, outcsv, args.n_proc_merge)
    print(f'Finish STEP 3.')
    print(f'Cost time for STEP 3: {time.time() - time_start:.2f}s')

    # === Step 4 ===
    print(f"\n===== STEP 4: Generating dataset .pt file =====")
    time_start = time.time()
    run_generate_dataset(args.target_path, args.input_pairs_path, output_json, outcsv, outpt, args.n_proc_merge)
    print(f'Finish STEP 4.')
    print(f'Cost time for STEP 4: {time.time() - time_start:.2f}s')


if __name__ == '__main__':
    main()

