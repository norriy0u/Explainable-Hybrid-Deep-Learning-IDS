# Dataset: UNSW-NB15

This directory contains the UNSW-NB15 dataset files required to run the Explainable Hybrid IDS.

> **The dataset CSV files are NOT included in this repository** due to their size. You must download them separately and place them in this folder.

---

## Required Files

Place the following files directly in this `dataset/` directory:

| Filename | Records | Purpose |
|---|---|---|
| `UNSW_NB15_training-set.csv` | 82,332 | Model training (fitted preprocessors, class weights) |
| `UNSW_NB15_testing-set.csv` | 175,341 | Model evaluation (never used during training) |

The file `NUSW-NB15_features.csv` (feature specification sheet) is already included.

---

## Download Instructions

### Option 1 — Official UNSW Research Website (Recommended)

1. Visit: **https://research.unsw.edu.au/projects/unsw-nb15-dataset**
2. Download the pre-split CSV files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
3. Place both files in this `dataset/` directory.

### Option 2 — Kaggle

1. Visit: **https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15**
2. Download the dataset archive.
3. Extract and copy the two CSV files here.

---

## Dataset Overview

The **UNSW-NB15** dataset was created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra. It contains real modern network traffic combined with synthetically generated attack traffic.

| Property | Value |
|---|---|
| Total Records | 257,673 |
| Training Records | 82,332 |
| Testing Records | 175,341 |
| Features | 42 numerical + 3 categorical (after removing identifiers) |
| Task Type | 10-class multiclass classification |

---

## Feature Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | Record identifier (excluded from model) |
| `attack_cat` | str | **Target label** — attack category name |
| `label` | int | Binary label (0=Normal, 1=Attack; excluded from model) |
| `proto` | str | Network protocol (categorical; encoded) |
| `service` | str | Destination service (categorical; encoded) |
| `state` | str | Connection state (categorical; encoded) |
| *(42 others)* | float/int | Numerical network flow features |

---

## 10 Classification Categories

| ID | Category | Description |
|---|---|---|
| 0 | Normal | Legitimate network traffic |
| 1 | Analysis | Port scans and vulnerability probing |
| 2 | Backdoor | Covert communication channels |
| **3** | **DoS** ⭐ | **Denial of Service (primary demo attack)** |
| 4 | Exploits | Known software vulnerability exploitation |
| 5 | Fuzzers | Randomized boundary-condition probing |
| 6 | Generic | Protocol-level cipher violations |
| 7 | Reconnaissance | Active network topology mapping |
| 8 | Shellcode | Executable shell injection payloads |
| 9 | Worms | Self-propagating malware replication |

---

## Citation

If you use the UNSW-NB15 dataset in your research, please cite:

```bibtex
@inproceedings{moustafa2015unsw,
  title     = {UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author    = {Moustafa, Nour and Slay, Jill},
  booktitle = {2015 Military Communications and Information Systems Conference (MilCIS)},
  pages     = {1--6},
  year      = {2015},
  organization = {IEEE}
}
```

---

## License & Usage

The UNSW-NB15 dataset is made available by UNSW Canberra for research and educational purposes. Please review the terms on the official website before use.
