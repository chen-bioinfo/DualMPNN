# DualMPNN
This is the official implement of **DualMPNN**, harnessing structural alignment templates for protein sequence recovery using Dual-stream MPNNs.

<img width="1157" height="690" alt="image" src="https://github.com/user-attachments/assets/6dfbbc15-9f64-4252-bd19-0ba91acfeff3" />

Paper url: https://neurips.cc/virtual/2025/loc/san-diego/poster/118062
# Setup
## 1. Setup Environment
a. You need to download foldseek implement locally and put it in foldseek diretory. It can be downloaded at https://github.com/steineggerlab/foldseek/releases

You should put the executable file in diretory of DualMPNN below.

    📁 foldseek
    └─ 📁 bin
        └─ 📄 foldseek (This is an executable file, about 700MB)

   
b. Install the conda environment by the following commands:

    conda create -n DualMPNN python=3.9 numpy=1.26
    conda activate DualMPNN
    pip install -r requirements.txt

## 2. Setup foldseek
Enter the directory and download the template dataset from foldseek server:    
    
    cd foldseek
    bin/foldseek databases PDB pdb tmp 

After downloading, process the dataset using this command:

    bin/foldseek convert2pdb pdb PDBdb --pdb-output-mode 1

After this command, the foldseek is successfully setup in your environment.

The detailed information about foldseek commands please visit the official repo: https://github.com/steineggerlab/foldseek

## 3. Find Templates
The model takes constructed format as input. Given your dataset directory path, you could generate the formatted dataset by running the script 
**template/findTemplate.py**.
This script will automatically find template using foldseek and generate .pt format file which can be directly utilized by train or test code.

**See findTemplate.py for detailed usage.**

You only need to generate the .pt format dataset once, unless you want to find different templates.
## 4. Train and Test
Run Dual_train.py script to train the model.

Run Dual_test.py script to test the model.
    
# Citation

```bibtex
@inproceedings{liaodualmpnn,
  title={DualMPNN: Harnessing Structural Alignments for High-Recovery Inverse Protein Folding},
  author={Liao Xuhui and Wang Qiyu and Liang Zhiqiang and others},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://neurips.cc/virtual/2025/loc/san-diego/poster/118062}
}
```



