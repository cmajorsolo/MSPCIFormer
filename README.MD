# MSPCIFormer
MSPCIFormer is a research project focused on advanced transformer architectures for multi-scale processing in deep learning. This repository contains code, experiments.

## Features
- Multi-scale transformer model implementation
- Modular and extensible codebase
- Training and evaluation scripts
- Experiment tracking and reproducibility

## Installation

```bash
git clone https://github.com/yourusername/mspciformer.git
cd mspciformer
pip install -r requirements.txt
```

## Usage
Train the model use run_longExp.py:
Example:
```!python run_longExp.py --model MSPCIFormer --seq_len 60 --pred_len 45  --d_model 32 --e_layers 2 --learning_rate 0.007 --dropout 0.31 --batch_size 8  --itr 1 --train_epochs 30  --patience 5 --lradj 3```

## Project Structure

- `models/` - Model definitions
- `data/` - Data loading and preprocessing

## Acknowledgement
This project builds upon and incorporates components from the following open-source repositories. We are grateful for their contributions to the time-series forecasting community:

- **PatchTST**: https://github.com/yuqinie98/PatchTST  
- **MSGNet**: https://github.com/YoZhibo/MSGNet  
- **iTransformer**: https://github.com/thuml/iTransformer  

Some parts of the implementation (e.g., data processing utilities, model modules, and training logic) are adapted or extended from these projects.