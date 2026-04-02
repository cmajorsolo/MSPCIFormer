"""
Optuna hyperparameter tuning for MSPCIFormer repo models.

Usage:
    python tuning/tune_hyperparams.py --model MSPCIFormer --n_trials 50
    python tuning/tune_hyperparams.py --model NBeats --n_trials 30 --nbeats_type generic
    python tuning/tune_hyperparams.py --model DLinear --n_trials 20

Results are saved to:
    tuning/results/{model}_best_params.json
    tuning/results/{model}_study.pkl   (full Optuna study, loadable for analysis)
"""

import sys
import os
import json
import argparse
import copy
import types

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Make sure repo root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp.exp_main import Exp_Main


# ---------------------------------------------------------------------------
# Default base args — mirrors run_longExp.py defaults
# ---------------------------------------------------------------------------

def get_base_args():
    args = types.SimpleNamespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id='tune',
        model='MSPCIFormer',
        data='custom',
        root_path='./data/',
        data_path='crypto_prices_wide.csv',
        features='MS',
        target='ada_close',
        freq='d',
        checkpoints='./tuning/checkpoints/',
        seq_len=5,
        label_len=1,
        pred_len=1,
        seasonal_patterns='Monthly',
        # architecture
        enc_in=8,
        dec_in=8,
        c_out=8,
        n_vars=8,
        d_model=32,
        n_heads=2,
        e_layers=2,
        d_layers=2,
        d_ff=64,
        dropout=0.33,
        head_dropout=0.0,
        embed='timeF',
        activation='gelu',
        factor=1,
        distil=True,
        output_attention=False,
        do_predict=False,
        moving_avg=25,
        embed_type=0,
        # patching
        patch_len=3,
        stride=2,
        padding_patch='end',
        revin=1,   # fixed: always enabled

        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=5,
        individual=0,
        # TimesNet / MSGNet
        top_k=2,
        num_kernels=6,
        num_class=1,
        num_nodes=7,
        subgraph_size=3,
        tanhalpha=3,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        # NBeats
        nbeats_type='interpretable',
        nbeats_trend_blocks=3,
        nbeats_trend_layers=4,
        nbeats_trend_layer_size=256,
        nbeats_seasonality_blocks=3,
        nbeats_seasonality_layers=4,
        nbeats_seasonality_layer_size=256,
        nbeats_degree_of_polynomial=3,
        nbeats_num_of_harmonics=1,
        nbeats_stacks=1,
        nbeats_layers=4,
        nbeats_layer_size=256,
        # MSPCIFormer / RAG
        chunk_size=3,
        k_retrieval=1,
        use_rag=0,
        n_cross_layers=1,
        if_fft=False,
        scale_list_not_fft=[4, 5],
        # iTransformer / TimeXer
        use_norm=True,   # fixed: always enabled
        # training
        num_workers=0,
        itr=1,
        train_epochs=5,
        batch_size=32,
        patience=3,
        learning_rate=0.005,
        des='tune',
        loss='MSE',
        lradj='TST',
        pct_start=0.3,
        use_amp=False,
        # GPU
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        device_ids=[0],
        test_flop=False,
        # walk-forward (disabled during tuning)
        walk_forward=False,
        wf_borders=None,
    )
    args.use_gpu = True if torch.cuda.is_available() else False
    return args


# ---------------------------------------------------------------------------
# Per-model search spaces
# ---------------------------------------------------------------------------

SHARED_SPACE = {
    'seq_len':       ('categorical', [3, 5, 10]),
    'learning_rate': ('float_log',   1e-4, 1e-2),
    'batch_size':    ('categorical', [16, 32, 64]),
    'dropout':       ('float',       0.1, 0.5),
}

MODEL_SPACES = {
    'MSPCIFormer': {
        **SHARED_SPACE,
        'd_model':            ('categorical', [32, 64, 128]),
        'n_heads':            ('categorical', [2, 4, 8]),
        'e_layers':           ('int',         1, 4),
        'd_ff':               ('categorical', [64, 128, 256]),
        'head_dropout':       ('float',       0.0, 0.5),
        'affine':             ('categorical', [0, 1]),
        'scale_list_not_fft': ('categorical', [[4, 5], [4, 8], [4, 8, 16]]),
        'stride':             ('categorical', [2, 3, 5, 16]),
        'top_k':              ('int',         2, 4),
    },
    'TimeXer': {
        **SHARED_SPACE,
        'd_model':      ('categorical', [32, 64, 128]),
        'n_heads':      ('categorical', [2, 4, 8]),
        'e_layers':     ('int',         1, 4),
        'd_ff':         ('categorical', [64, 128, 256]),
        'head_dropout': ('float',       0.0, 0.5),
        'patch_len':    ('categorical', [3, 4, 5, 7]),
        'factor':       ('categorical', [1, 3, 5]),
    },
    'iTransformer': {
        **SHARED_SPACE,
        'd_model':  ('categorical', [32, 64, 128]),
        'n_heads':  ('categorical', [2, 4, 8]),
        'e_layers': ('int',         1, 4),
        'd_ff':     ('categorical', [64, 128, 256]),
        'factor':   ('categorical', [1, 3, 5]),
    },
    'TimesNet': {
        **SHARED_SPACE,
        'd_model':    ('categorical', [32, 64, 128]),
        'd_ff':       ('categorical', [64, 128, 256]),
        'e_layers':   ('int',         1, 4),
        'top_k':      ('int',         1, 4),
        'num_kernels':('int',         3, 8),
    },
    'MSGNet': {
        **SHARED_SPACE,
        'd_model':     ('categorical', [32, 64, 128]),
        'd_ff':        ('categorical', [64, 128, 256]),
        'e_layers':    ('int',         1, 3),
        'top_k':       ('int',         1, 4),
        'conv_channel':('categorical', [16, 32, 64]),
        'skip_channel':('categorical', [16, 32, 64]),
        'gcn_depth':   ('int',         1, 3),
        'propalpha':   ('float',       0.05, 0.5),
        'node_dim':    ('categorical', [8, 16, 32]),
    },
    'DLinear': {
        **SHARED_SPACE,
        'moving_avg':  ('categorical', [5, 13, 25]),
        'kernel_size': ('categorical', [3, 5, 7, 11, 15]),
        'individual':  ('categorical', [0, 1]),
    },
    'NBeats_interpretable': {
        **SHARED_SPACE,
        'nbeats_trend_blocks':          ('int',         1, 4),
        'nbeats_trend_layers':          ('int',         2, 6),
        'nbeats_trend_layer_size':      ('categorical', [64, 128, 256, 512]),
        'nbeats_seasonality_blocks':    ('int',         1, 4),
        'nbeats_seasonality_layers':    ('int',         2, 6),
        'nbeats_seasonality_layer_size':('categorical', [64, 128, 256, 512]),
        'nbeats_degree_of_polynomial':  ('int',         2, 5),
        'nbeats_num_of_harmonics':      ('int',         1, 4),
    },
    'NBeats_generic': {
        **SHARED_SPACE,
        'nbeats_stacks':     ('int',         1, 4),
        'nbeats_layers':     ('int',         2, 6),
        'nbeats_layer_size': ('categorical', [64, 128, 256, 512]),
    },
}


def suggest_params(trial, space):
    """Sample hyperparameters from the search space for this trial."""
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == 'categorical':
            params[name] = trial.suggest_categorical(name, spec[1])
        elif kind == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == 'float':
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == 'float_log':
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
    return params


# ---------------------------------------------------------------------------
# Constraint: d_model must be divisible by n_heads for attention models
# ---------------------------------------------------------------------------

def fix_attention_constraint(params):
    if 'd_model' in params and 'n_heads' in params:
        while params['d_model'] % params['n_heads'] != 0:
            params['n_heads'] = params['n_heads'] // 2
            if params['n_heads'] < 1:
                params['n_heads'] = 1
                break
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(base_args, model_key, tune_model_name):
    space = MODEL_SPACES[model_key]

    def objective(trial):
        args = copy.copy(base_args)
        args.model = tune_model_name

        # NBeats: set type from model_key
        if tune_model_name == 'NBeats':
            args.nbeats_type = 'interpretable' if 'interpretable' in model_key else 'generic'

        params = suggest_params(trial, space)
        params = fix_attention_constraint(params)

        for k, v in params.items():
            setattr(args, k, v)

        # label_len must be <= seq_len
        args.label_len = min(args.label_len, args.seq_len)

        setting = 'tune_{}_trial{}'.format(model_key, trial.number)

        try:
            exp = Exp_Main(args)
            _, best_vali_loss = exp.train(setting)
        except Exception as e:
            print(f'Trial {trial.number} failed: {e}')
            raise optuna.exceptions.TrialPruned()

        # Clean up checkpoint to save disk space
        ckpt_dir = os.path.join(args.checkpoints, setting)
        if os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)

        return best_vali_loss

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')
    parser.add_argument('--model', type=str, default='DLinear',
                        choices=['MSPCIFormer', 'TimeXer', 'MSGNet', 'TimesNet',
                                 'NBeats', 'iTransformer', 'DLinear'],
                        help='model to tune')
    parser.add_argument('--nbeats_type', type=str, default='interpretable',
                        choices=['interpretable', 'generic'],
                        help='NBeats type (only used when --model NBeats)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='parallel trials (1 = sequential, safe default)')
    parser.add_argument('--data_path', type=str, default='crypto_prices_wide.csv',
                        help='data file name inside root_path')
    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='epochs per trial (keep low for speed)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='verify code only: forces n_trials=1, train_epochs=1, batch_size=64, num_workers=0, patience=1')

    args_tune = parser.parse_args()

    if args_tune.dry_run:
        args_tune.n_trials     = 1
        args_tune.train_epochs = 1
        print('[dry_run] n_trials=1, train_epochs=1, batch_size=64, num_workers=0, patience=1')

    # Resolve NBeats model key
    if args_tune.model == 'NBeats':
        model_key = f'NBeats_{args_tune.nbeats_type}'
    else:
        model_key = args_tune.model

    base_args = get_base_args()
    base_args.data_path = args_tune.data_path
    base_args.root_path = args_tune.root_path
    base_args.train_epochs = args_tune.train_epochs
    base_args.model = args_tune.model

    if args_tune.dry_run:
        base_args.batch_size   = 64
        base_args.num_workers  = 0
        base_args.patience     = 1

    os.makedirs('./tuning/results', exist_ok=True)
    os.makedirs('./tuning/checkpoints', exist_ok=True)

    # --- Pre-run summary ---
    space = MODEL_SPACES[model_key]
    print('\n' + '=' * 60)
    print('Optuna hyperparameter search')
    print(f'  Model      : {args_tune.model}' + (f' ({args_tune.nbeats_type})' if args_tune.model == 'NBeats' else ''))
    print(f'  Model key  : {model_key}')
    print(f'  n_trials   : {args_tune.n_trials}')
    print(f'  train_epochs per trial: {args_tune.train_epochs}')
    print(f'  dry_run    : {args_tune.dry_run}')
    print(f'  data_path  : {args_tune.data_path}')
    print('Search space:')
    for param, spec in space.items():
        kind = spec[0]
        if kind == 'categorical':
            print(f'  {param:<35s} categorical  {spec[1]}')
        elif kind == 'int':
            print(f'  {param:<35s} int          [{spec[1]}, {spec[2]}]')
        elif kind == 'float':
            print(f'  {param:<35s} float        [{spec[1]}, {spec[2]}]')
        elif kind == 'float_log':
            print(f'  {param:<35s} float (log)  [{spec[1]}, {spec[2]}]')
    print('=' * 60 + '\n')

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=model_key,
    )

    objective = make_objective(base_args, model_key, args_tune.model)
    study.optimize(objective, n_trials=args_tune.n_trials, n_jobs=args_tune.n_jobs)

    # Save results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print('\nNo trials completed successfully. Check errors above.')
        return

    best_params = study.best_params
    best_value  = study.best_value

    print(f'\n===== Best params for {model_key} =====')
    print(f'Best validation loss: {best_value:.7f}')
    for k, v in best_params.items():
        print(f'  {k}: {v}')

    out = {
        'model': args_tune.model,
        'model_key': model_key,
        'best_vali_loss': best_value,
        'best_params': best_params,
    }
    out_path = f'./tuning/results/{model_key}_best_params.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to {out_path}')

    # Save full study for later analysis (importance plots etc.)
    import pickle
    study_path = f'./tuning/results/{model_key}_study.pkl'
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f'Full study saved to {study_path}')


if __name__ == '__main__':
    main()
