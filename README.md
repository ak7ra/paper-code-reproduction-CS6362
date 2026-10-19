# Paper Code Reproduction — CS6362

This repository contains code, Jupyter notebooks, and experimental outputs for reproducing selected results from three machine learning research papers as part of CS6362 (Advanced Machine Learning).

## Papers and Reproduction Targets

### Paper 1: Understanding Deep Learning Requires Rethinking Generalization

**Authors:** Zhang et al.  
**Reproduction target:** Figures 1(a–c)

**Main files and directories:**

- `paper1_Zhang_etal/notebooks/paper1.py` — Python implementation.
- `paper1_Zhang_etal/notebooks/reproduction.ipynb` — Jupyter notebook.
- `paper1_Zhang_etal/notebooks/test_results/` — Preliminary experimental results.
- `paper1_Zhang_etal/notebooks/test_results_5000/` — Additional experimental results.

### Paper 2: On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima

**Authors:** Keskar et al.  
**Reproduction target:** Figure 3

**Main files and directories:**

- `paper2_Keskar_etal/notebooks/keskar_fig3.py` — Python implementation for Figure 3.
- `paper2_Keskar_etal/notebooks/reproduction2.ipynb` — Jupyter notebook.
- `paper2_Keskar_etal/notebooks/results/F1/` — Committed experimental outputs for the F1 network.

### Paper 3: β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework

**Authors:** Higgins et al.  
**Reproduction targets:** Figures 7A and 7B

**Main files and directories:**

- `paper3_Higgens_etal/reproduce_fig7.py` — Python implementation for reproducing Figure 7.
- `paper3_Higgens_etal/reproduction3.ipynb` — Jupyter notebook.
- `paper3_Higgens_etal/fig7_outputs/` — Generated figures and summary data.

## Repository Structure

The following tree summarizes the documented repository organization. Some intermediate directories are shown without listing every file.

```text
paper-code-reproduction-CS6362/
├── README.md
├── .gitignore
│
├── paper1_Zhang_etal/
│   ├── notebooks/
│   │   ├── paper1.py
│   │   ├── reproduction.ipynb
│   │   ├── test_results/
│   │   │   └── runs/
│   │   └── test_results_5000/
│   │       └── runs/
│   └── GOOGLE COLAB FILES/
│       └── Paper1.ipynb
        └── paper1_results/
│           ├── runs/
│           ├── fig1a.csv
│           ├── fig1a_reproduced.png
│           ├── fig1b.csv
│           ├── fig1b_reproduced.png
│           ├── fig1c.csv
│           └── fig1c_reproduced.png
│
├── paper2_Keskar_etal/
│   ├── notebooks/
│   │   ├── keskar_fig3.py
│   │   ├── reproduction2.ipynb
│   │   └── results/
│   │       └── F1/
│   │           ├── fig3a_F1_trial4_reestimate.pdf
│   │           ├── fig3a_F1_trial4_reestimate.png
│   │           ├── interp_trial4_reestimate.csv
│   │           ├── interp_trial4_reestimate_meta.json
│   │           └── table2_summary.json
│   └── GOOGLE COLAB FILES/
│       └── Paper2.ipynb
        └── paper2_results/
            ├── c1_full/
            │   └── C1/
            │       ├── fig3c_C1_trial4_reestimate.pdf
            │       ├── fig3c_C1_trial4_reestimate.png
            │       ├── interp_trial4_reestimate.csv
            │       ├── interp_trial4_reestimate_meta.json
            │       └── table2_summary.json
            │
            └── c2_limited/
                └── C2/
                    ├── fig3c_C1_trial4_reestimate.pdf
                    ├── fig3c_C1_trial4_reestimate.png
                    ├── interp_trial4_reestimate.csv
                    ├── interp_trial4_reestimate_meta.json
                    └── table2_summary.json
│
└── paper3_Higgens_etal/
    ├── reproduce_fig7.py
    ├── reproduction3.ipynb
    └── fig7_outputs/
        ├── figure7A.png
        ├── figure7B.png
        ├── figure7AB_combined.png
        └── summary.json
```

## Environment and Setup

The experiments were conducted using local computing resources and Google Colab. Some runs used a local CPU, while others used an NVIDIA T4 GPU in Google Colab.

A single environment configuration is not documented here because the experiments may have different dependencies and hardware requirements.

## Use of Generative AI

Generative AI tools were used to assist with understanding the papers, planning the reproductions, generating and modifying code, and troubleshooting implementations.

The code consists of locally maintained implementations developed with AI assistance. It is not presented as the original authors' implementation code.

## Disclaimer

This repository is maintained for coursework and educational research purposes. The reproduced implementations and figures are not official implementations or results from the original paper authors.
