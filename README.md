# DualMPNN
This is the official implement of **DualMPNN**, which harnesses structural alignment templates for protein sequence recovery using Dual-stream MPNNs.

# Setup
## 1. Setup Environment
a. You need to download foldseek implement locally and put it in foldseek diretory. It can be downloaded at https://github.com/steineggerlab/foldseek/releases

    📁 foldseek
    └─ 📁 bin
        └─ 📄 foldseek (This is an executable file)

   
b. Install the conda environment by the following commands:

    conda create -n DualMPNN python=3.9 numpy=1.26
    conda activate DualMPNN
    pip install -r requirements.txt

## 2. Setup foldseek
Enter the directory and download the template dataset from foldseek server:    
    
    cd foldseek
    bin/foldseek databases PDB pdb tmp 

The detailed information about foldseek commands please visit the official repo: https://github.com/steineggerlab/foldseek

## 3. Find Templates
The model takes constructed format as input. You could generate the dataset by running the script **template/findTemplate.py
**. This script will automatically find template using foldseek and generate .pt format file which can be directly utilized by train or test code.
    




