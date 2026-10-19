# Paper Code Reproduction — CS6362

This repository contains code, Jupyter notebooks, and experimental outputs for reproducing selected results from three machine learning research papers as part of CS6362 (Advanced Machine Learning).

## Papers

### Paper 1: Understanding Deep Learning Requires Rethinking Generalization

**Authors:** Zhang et al.

**Reproduction target:** Figures 1(a–c)

This experiment investigates the ability of neural networks to fit training data under varying levels of label corruption.

The implementation explores:
- Label corruption rates ranging from 0 to 1.
- Different neural network architectures, including Inception and multilayer perceptrons.
- Training convergence under different corruption levels.
- Changes in training and test performance.

**Files:**
- `paper1_Zhang_etal/notebooks/paper1.py` — Python implementation.
- `paper1_Zhang_etal/notebooks/reproduction.ipynb` — Jupyter notebook.
- `paper1_Zhang_etal/notebooks/test_results/` — Preliminary experimental results.
- `paper1_Zhang_etal/notebooks/test_results_5000/` — Additional experimental results.

The committed JSON files contain selected runs rather than a complete experimental sweep. Final results are included in the "paper1_Zhang_etal/GOOGLE COLAB" folder.

### Paper 2: On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima

**Authors:** Keskar et al.

**Reproduction target:** Figure 3

This experiment examines differences between neural networks trained using small and large mini-batches, particularly the sharpness of their loss landscapes.

The implementation explores:
- Small-batch and large-batch training.
- Interpolation between trained model parameters.
- Changes in loss along interpolation paths.
- The relationship between batch size, loss landscape geometry, and generalization.

**Files:**
- `paper2_Keskar_etal/notebooks/keskar_fig3.py` — Python implementation.
- `paper2_Keskar_etal/notebooks/reproduction2.ipynb` — Jupyter notebook.
- `paper2_Keskar_etal/notebooks/results/F1/` — Experimental outputs for the F1 network.

The committed F1 outputs include an interpolation plot, numerical interpolation results, metadata, and summary statistics. Results for C1 and C2 are included in the "paper2_Keskar_etal/GOOGLE COLAB" folder.

### Paper 3: β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework

**Authors:** Higgens et al.

**Reproduction target:** Figures 7A and 7B

This experiment investigates aspects of the β-VAE framework, which extends the variational autoencoder objective by introducing a weighting coefficient on the KL-divergence term.

**Files:**
- `paper3_Higgens_etal/reproduce_fig7.py` — Python implementation.
- `paper3_Higgens_etal/reproduction3.ipynb` — Jupyter notebook.
- `paper3_Higgens_etal/fig7_outputs/` — Generated figures and summary data.

The committed outputs include reproductions labeled Figure 7A and Figure 7B, a combined visualization, and a JSON summary.

## Repository Structure

```text
paper-code-reproduction-CS6362/
├── README.md
├── .gitignore
│
├── paper1_Zhang_etal/
│   └── notebooks/
│       ├── paper1.py
│       ├── reproduction.ipynb
│       ├── test_results/
│       │   └── runs/
│       └── test_results_5000/
│           └── runs/
│
├── paper2_Keskar_etal/
│   └── notebooks/
│       ├── keskar_fig3.py
│       ├── reproduction2.ipynb
│       └── results/
│           └── F1/
│               ├── fig3a_F1_trial4_reestimate.pdf
│               ├── fig3a_F1_trial4_reestimate.png
│               ├── interp_trial4_reestimate.csv
│               ├── interp_trial4_reestimate_meta.json
│               └── table2_summary.json
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

## Reproduction Scope and Limitations

These implementations are educational reproductions of selected experiments rather than complete reimplementations of the original papers.

Several limitations apply:

- Only selected figures and experimental configurations are targeted.
- Some experiments use reduced computational budgets.
- The available hardware may differ substantially from that used in the original publications.
- Certain experiments have only partial results committed to the repository.
- Results may differ from the original publications because of implementation choices, random initialization, training budgets, or other experimental conditions.

The committed outputs document the available local experiments and should not be interpreted as verification of all results in the original papers.

Experiments were conducted using both local computing resources and Google Colab. Some experiments were executed on a local CPU, while others utilized Google Colab's NVIDIA T4 GPU. Experimental outputs generated in Google Colab are organized in the corresponding GOOGLE COLAB directories. Due to differences in computational resources and runtime constraints, certain experiments were conducted with reduced training budgets or modified configurations.

## Use of Generative AI

Generative AI tools were used to assist with understanding the papers, planning the reproductions, generating and modifying code, and troubleshooting implementations.

The code in this repository consists of locally maintained implementations developed with AI assistance. It is not presented as replicas of the original authors' implementation code.


## Disclaimer

This repository is maintained for coursework and educational research purposes. The reproduced implementations and figures are not official implementations or results from the original paper authors.