"""
Multi-horizon model comparison experiment.

Trains and evaluates all models across prediction horizons {1, 3, 5, 7, 15, 30},
using best hyperparameters from Optuna tuning results.

Runs both hold-out (70/10/20) and walk-forward (expanding window) validation.
Tracks training time per epoch and peak memory usage.

Results saved to: comparison/comparison_results.csv

Usage:
    python run_comparison.py
    python run_comparison.py --pred_lens 1 3 5 7
    python run_comparison.py --models MSPCIFormer DLinear --n_folds 3
    python run_comparison.py --dry_run
"""

import argparse
import copy
import csv
import json
import os
import random
import types

import numpy as np
import pandas as pd
import torch

from exp.exp_main import Exp_Main


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_MODELS = [
    'DLinear',
    'NBeats_interpretable',
    'NBeats_generic',
    'TimesNet',
    'MSGNet',
    'iTransformer',
    'TimeXer',
    'MSPCIFormer',
]

DEFAULT_PRED_LENS = [1, 3, 5, 7, 15, 30]

RESULTS_DIR = './comparison'
RESULTS_CSV = os.path.join(RESULTS_DIR, 'comparison_results.csv')
EPOCH_TIMES_CSV = os.path.join(RESULTS_DIR, 'epoch_times.csv')

CSV_COLUMNS = [
    'model', 'pred_len', 'validation_type', 'fold',
    'mse', 'mae', 'rmse', 'mape', 'mspe', 'rse', 'nd', 'nrmse',
    'mda', 'sharpe', 'max_drawdown',
    'avg_time_per_epoch_s', 'peak_memory_mb',
]

EPOCH_TIME_COLUMNS = ['model', 'pred_len', 'validation_type', 'fold', 'epoch', 'time_s']


# ---------------------------------------------------------------------------
# Base args (mirrors tune_hyperparams.py defaults)
# ---------------------------------------------------------------------------

def get_base_args():
    args = types.SimpleNamespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id='comparison',
        model='DLinear',
        data='custom',
        root_path='./data/',
        data_path='crypto_prices_wide.csv',
        features='MS',
        target='ada_close',
        freq='d',
        checkpoints='./comparison/checkpoints/',
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
        revin=1,
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
        use_norm=True,
        # training
        num_workers=0,
        itr=1,
        train_epochs=30,
        batch_size=32,
        patience=5,
        learning_rate=0.005,
        des='comparison',
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
        # walk-forward (set per run)
        walk_forward=False,
        wf_borders=None,
    )
    args.use_gpu = torch.cuda.is_available()
    return args


# ---------------------------------------------------------------------------
# Load best params from Optuna JSON
# ---------------------------------------------------------------------------

def load_best_params(model: str, tuning_dir: str = './tuning/results') -> dict:
    """Load best hyperparameters from Optuna tuning results JSON."""
    # NBeats_interpretable / NBeats_generic map directly to their JSON keys
    model_key = model
    path = os.path.join(tuning_dir, f'{model_key}_best_params.json')
    if not os.path.exists(path):
        print(f'  [warn] No tuning results for {model} at {path}. Using defaults.')
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {})


def apply_params(args, params: dict):
    """Apply a dict of hyperparameters to args namespace."""
    for k, v in params.items():
        setattr(args, k, v)
    # Ensure label_len <= seq_len
    args.label_len = min(args.label_len, args.seq_len)
    return args


# ---------------------------------------------------------------------------
# Single run: train + test, return metrics row
# ---------------------------------------------------------------------------

def run_single(args, setting: str, epoch_writer, epoch_csvfile,
               model: str, pred_len: int, validation_type: str, fold: int):
    """Train and test one experiment. Returns dict of metrics."""
    exp = Exp_Main(args)
    _, best_vali_loss, avg_time_per_epoch, epoch_times = exp.train(setting)

    # Write one row per epoch to epoch_times.csv
    for epoch_idx, t in enumerate(epoch_times):
        epoch_writer.writerow([model, pred_len, validation_type, fold, epoch_idx + 1, round(t, 4)])
    epoch_csvfile.flush()

    mse, mae, rmse, mape, mspe, rse, nd, nrmse, mda, sharpe, max_dd, peak_memory_mb = exp.test(setting)
    torch.cuda.empty_cache()
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'mspe': float(mspe),
        'rse': float(rse),
        'nd': float(nd),
        'nrmse': float(nrmse),
        'mda': float(mda),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'avg_time_per_epoch_s': float(avg_time_per_epoch),
        'peak_memory_mb': float(peak_memory_mb),
    }


# ---------------------------------------------------------------------------
# Walk-forward helper
# ---------------------------------------------------------------------------

def run_walk_forward(args, model: str, pred_len: int, n_folds: int,
                     fold_size: float, total_len: int, writer, csvfile,
                     epoch_writer, epoch_csvfile):
    """Run walk-forward CV and write one row per fold."""
    val_size = int(total_len * 0.10)
    fold_step = int(total_len * fold_size / n_folds)
    initial_train = int(total_len * fold_size)
    min_test_rows = args.seq_len + pred_len + 1

    folds_run = 0
    for fold in range(n_folds):
        train_end = initial_train + fold * fold_step
        val_end = train_end + val_size
        test_end = min(val_end + fold_step, total_len)

        if val_end >= total_len or (test_end - val_end) < min_test_rows:
            print(f'    Fold {fold}: window too small, stopping early.')
            break

        args_fold = copy.copy(args)
        args_fold.wf_borders = {'train_end': train_end, 'val_end': val_end, 'test_end': test_end}

        setting = f'cmp_{model}_predl{pred_len}_wf_fold{fold}'
        print(f'    WF fold {fold}: train[0,{train_end}) val[{train_end},{val_end}) test[{val_end},{test_end})')

        try:
            metrics = run_single(args_fold, setting, epoch_writer, epoch_csvfile,
                                 model, pred_len, 'walk_forward', fold)
        except Exception as e:
            print(f'    [error] fold {fold} failed: {e}')
            continue

        row = {'model': model, 'pred_len': pred_len, 'validation_type': 'walk_forward',
               'fold': fold, **metrics}
        writer.writerow([row[c] for c in CSV_COLUMNS])
        csvfile.flush()
        folds_run += 1

    return folds_run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-horizon model comparison')
    parser.add_argument('--models', type=str, nargs='+', default=ALL_MODELS,
                        help='models to compare')
    parser.add_argument('--pred_lens', type=int, nargs='+', default=DEFAULT_PRED_LENS,
                        help='prediction horizons to evaluate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='walk-forward folds')
    parser.add_argument('--fold_size', type=float, default=0.5,
                        help='initial training window fraction for walk-forward')
    parser.add_argument('--tuning_dir', type=str, default='./tuning/results',
                        help='directory containing Optuna best_params JSON files')
    parser.add_argument('--skip_walk_forward', action='store_true', default=False,
                        help='only run hold-out validation (faster)')
    parser.add_argument('--skip_hold_out', action='store_true', default=False,
                        help='only run walk-forward validation')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quick smoke test: 1 epoch, 1 WF fold, pred_lens=[1]')
    cli = parser.parse_args()

    if cli.dry_run:
        cli.train_epochs = 2
        cli.n_folds = 2
        cli.pred_lens = [1]
        print('[dry_run] train_epochs=1, n_folds=1, pred_lens=[1]')

    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'checkpoints'), exist_ok=True)

    # Determine total dataset length
    base = get_base_args()
    df_raw = pd.read_csv(os.path.join(base.root_path, base.data_path))
    total_len = len(df_raw)
    print(f'Dataset length: {total_len} rows')

    # Load existing CSVs, drop rows for models being re-run, then rewrite.
    # This lets you run one model at a time without losing other models' results.
    KEY_COLS = ['model', 'pred_len', 'validation_type', 'fold']
    EPOCH_KEY_COLS = ['model', 'pred_len', 'validation_type', 'fold', 'epoch']

    def _load_existing(path, columns):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return pd.DataFrame(columns=columns)

    existing_results = _load_existing(RESULTS_CSV, CSV_COLUMNS)
    existing_epochs  = _load_existing(EPOCH_TIMES_CSV, EPOCH_TIME_COLUMNS)

    # Drop stale rows for models in this run (identified by model column)
    labels_being_run = set(cli.models)  # e.g. {'NBeats_interpretable', 'MSPCIFormer'}
    if not existing_results.empty:
        existing_results = existing_results[~existing_results['model'].isin(labels_being_run)]
    if not existing_epochs.empty:
        existing_epochs = existing_epochs[~existing_epochs['model'].isin(labels_being_run)]

    # Write filtered existing rows back + stream new rows as they come in
    csvfile = open(RESULTS_CSV, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(CSV_COLUMNS)
    if not existing_results.empty:
        for row in existing_results.itertuples(index=False):
            writer.writerow(list(row))
    csvfile.flush()

    epoch_csvfile = open(EPOCH_TIMES_CSV, 'w', newline='')
    epoch_writer = csv.writer(epoch_csvfile)
    epoch_writer.writerow(EPOCH_TIME_COLUMNS)
    if not existing_epochs.empty:
        for row in existing_epochs.itertuples(index=False):
            epoch_writer.writerow(list(row))
    epoch_csvfile.flush()

    total_runs = len(cli.models) * len(cli.pred_lens)
    run_idx = 0

    for model in cli.models:
        best_params = load_best_params(model, cli.tuning_dir)

        for pred_len in cli.pred_lens:
            run_idx += 1
            print(f'\n{"="*60}')
            print(f'[{run_idx}/{total_runs}] Model={model}  pred_len={pred_len}')
            print(f'{"="*60}')

            # Resolve actual model name and nbeats_type from the model key
            if model == 'NBeats_interpretable':
                model_name, nbeats_type = 'NBeats', 'interpretable'
            elif model == 'NBeats_generic':
                model_name, nbeats_type = 'NBeats', 'generic'
            else:
                model_name, nbeats_type = model, None

            # Build args for this (model, pred_len) combination
            args = get_base_args()
            args.model = model_name
            if nbeats_type:
                args.nbeats_type = nbeats_type
            args.pred_len = pred_len
            args.train_epochs = cli.train_epochs
            apply_params(args, best_params)
            # pred_len overrides any Optuna suggestion
            args.pred_len = pred_len

            # Print hyperparameters in use
            if best_params:
                print('  Hyperparameters (from Optuna tuning):')
                for k, v in best_params.items():
                    print(f'    {k}: {v}')
            else:
                print('  Hyperparameters: using defaults (no tuning results found)')
            print(f'  Overrides applied: pred_len={pred_len}, train_epochs={cli.train_epochs}')

            # --- Hold-out validation ---
            if not cli.skip_hold_out:
                args_ho = copy.copy(args)
                args_ho.wf_borders = None
                setting = f'cmp_{model}_predl{pred_len}_holdout'
                print(f'  Hold-out: {setting}')
                try:
                    metrics = run_single(args_ho, setting, epoch_writer, epoch_csvfile,
                                         model, pred_len, 'hold_out', -1)
                    row = {'model': model, 'pred_len': pred_len,
                           'validation_type': 'hold_out', 'fold': -1, **metrics}
                    writer.writerow([row[c] for c in CSV_COLUMNS])
                    csvfile.flush()
                    print(f'  Hold-out done | mse={metrics["mse"]:.6f}  mda={metrics["mda"]:.4f}  '
                          f'sharpe={metrics["sharpe"]:.4f}  t/epoch={metrics["avg_time_per_epoch_s"]:.1f}s  '
                          f'mem={metrics["peak_memory_mb"]:.1f}MB')
                except Exception as e:
                    print(f'  [error] hold-out failed: {e}')

            # --- Walk-forward validation ---
            if not cli.skip_walk_forward:
                args_wf = copy.copy(args)
                print(f'  Walk-forward: {cli.n_folds} folds, fold_size={cli.fold_size}')
                folds_run = run_walk_forward(
                    args_wf, model, pred_len,
                    n_folds=cli.n_folds,
                    fold_size=cli.fold_size,
                    total_len=total_len,
                    writer=writer,
                    csvfile=csvfile,
                    epoch_writer=epoch_writer,
                    epoch_csvfile=epoch_csvfile,
                )
                print(f'  Walk-forward done ({folds_run} folds completed)')

    csvfile.close()
    epoch_csvfile.close()
    print('\n' + '=' * 60)
    print('Comparison complete.')
    print(f'  Results      : {RESULTS_CSV}')
    print(f'  Epoch times  : {EPOCH_TIMES_CSV}')
    print('=' * 60)


if __name__ == '__main__':
    main()
