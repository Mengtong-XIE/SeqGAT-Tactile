# SeqGAT-Tactile

This repository contains two parts:
- `baseline/`: Baseline code adapted from Wen's implementation (with my configuration/scripts for reproduction).
- `seqgat/`: My own SeqGAT implementation (spatio-temporal graph attention for TacTip).

## Quick Start
```bash
# Baseline training (example)
cd baseline
pip install -r ../requirements.txt
python scripts/train.py --config configs/tactile_gat_k5_s.yaml

# SeqGAT training (example)
cd ../seqgat
pip install -r ../requirements.txt
python scripts/train.py --config configs/seqgat_k5_st.yaml
