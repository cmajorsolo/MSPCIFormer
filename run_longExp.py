import argparse
import os
import time
from multiprocessing import freeze_support
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pandas as pd

def main():   
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='MSGNet for Time Series Forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
    # required arguments: is_training, model_id, model, data; Change required to False for debugging from this script
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='NBeats', help='model id')
    parser.add_argument('--model', type=str, required=False, default='NBeats',
                        help='model name, options: '
                             '[Autoformer, Informer, Transformer, MSGNet, DLinear, TimeXer, TimesNet, NBeats, MSPCIFormer, PatchTST, iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='crypto_prices_wide.csv', help='data file')

    # RAG parameters
    parser.add_argument('--chunk_size', type=int, default=3, help='RAG chunk size (number of timesteps per chunk)')
    parser.add_argument('--k_retrieval', type=int, default=1, help='RAG number of historical chunks to retrieve')
    # MSPFormerCross parameters
    parser.add_argument('--use_rag', type=int, default=1, help='Whether to use RAG (1=yes, 0=no)')
    parser.add_argument('--n_cross_layers', type=int, default=1, help='Number of cross-attention layers for MSPFormerCross')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,'
                            ' S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ada_close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # ARIMAX specific hyper-parameters
    parser.add_argument('--ar_order', type=int, default=3, help='ARIMAX autoregressive order (p)')
    parser.add_argument('--diff_order', type=int, default=2, help='ARIMAX differencing order (d)')
    parser.add_argument('--ma_order', type=int, default=1, help='ARIMAX moving-average order (q)')

    parser.add_argument('--top_k', type=int, default=2, help='for TimesBlock/ScaleGraphBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

    parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
    parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
    parser.add_argument('--tanhalpha', type=float, default=3, help='')

    #GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    # NBeats
    parser.add_argument('--nbeats_type', type=str, default='interpretable', choices=['interpretable', 'generic'], help='NBeats model type')
    parser.add_argument('--nbeats_trend_blocks', type=int, default='3', help='NBeats trend blocks')
    parser.add_argument('--nbeats_trend_layers', type=int, default='4', help='NBeats trend layers')
    parser.add_argument('--nbeats_trend_layer_size', type=int, default='256', help='NBeats trend layer size')
    parser.add_argument('--nbeats_seasonality_blocks', type=int, default='3', help='NBeats seasonality blocks')
    parser.add_argument('--nbeats_seasonality_layers', type=int, default='4', help='NBeats seasonality layers')
    parser.add_argument('--nbeats_seasonality_layer_size', type=int, default='256', help='NBeats seasonality layer size')
    parser.add_argument('--nbeats_degree_of_polynomial', type=int, default='3', help='NBeats degree of polynomial for seasonality')    
    parser.add_argument('--nbeats_stacks', type=int, default='1', help='NBeats generic stacks')
    parser.add_argument('--nbeats_layers', type=int, default='4', help='NBeats generic layers')
    parser.add_argument('--nbeats_layer_size', type=int, default='256', help='NBeats generic layer size')

    # DLinear
    # parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    
    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=3, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=5, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0, help='0: default '
                                                                '1: value embedding + temporal embedding + positional embedding '
                                                                '2: value embedding + temporal embedding '
                                                                '3: value embedding + positional embedding '
                                                                '4: value embedding')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--n_vars', type=int, default=8, help='number of features or channels; for CDPatchTST')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.33, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    # iTransformer
    parser.add_argument('--use_norm', action='store_true', default=True, help='whether to use normalization')
    # MultiScalePatching
    parser.add_argument('--if_fft', action='store_true', default=False, help='enable FFT for choosing patching scales')
    parser.add_argument('--scale_list_not_fft', type=int, nargs='+', default=[4, 5], help='patching scales to choose from (used when --if_fft is not set)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Walk-forward validation
    parser.add_argument('--walk_forward', action='store_true', default=True, help='enable walk-forward validation')
    parser.add_argument('--n_folds', type=int, default=5, help='number of folds for walk-forward validation')
    parser.add_argument('--fold_size', type=float, default=0.5, help='fraction of total data used as the initial training window; each fold expands by fold_size/n_folds')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        start = time.time()

        if args.walk_forward:
            # --- Walk-forward validation (expanding window, n_folds folds) ---
            # Determine total dataset length from the CSV
            df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
            total_len = len(df_raw)

            # fold_size fraction controls the initial training window.
            # Each fold steps forward by (fold_size / n_folds) of total data.
            # Val is fixed at 10% of total; test is one fold-step.
            n_folds       = args.n_folds
            val_size      = int(total_len * 0.10)
            fold_step     = int(total_len * args.fold_size / n_folds)
            initial_train = int(total_len * args.fold_size)    # train size for fold 0

            fold_metrics = []
            metric_names = ['mse', 'mae', 'rmse', 'mape', 'mspe', 'rse', 'nd', 'nrmse', 'mda', 'sharpe', 'max_dd']

            for fold in range(n_folds):
                train_end = initial_train + fold * fold_step
                val_end   = train_end + val_size
                test_end  = val_end + fold_step
                test_end  = min(test_end, total_len)   # clamp last fold
                min_test_rows = args.seq_len + args.pred_len + 1
                if val_end >= total_len or (test_end - val_end) < min_test_rows:
                    print(f'Fold {fold}: test window too small ({test_end - val_end} rows, need {min_test_rows}), skipping.')
                    break

                args.wf_borders = {'train_end': train_end, 'val_end': val_end, 'test_end': test_end}

                setting = '{}_{}_data{}_feature{}_seql{}_labell{}_predl{}_dmodel{}_nheads{}_elayers{}_dlayers{}_dff{}_fc{}_embed{}_distil{}_{}_fold{}'.format(
                    args.model_id, args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers,
                    args.d_ff, args.factor, args.embed, args.distil, args.des, fold)

                print(f'\n>>>>>>>Walk-forward fold {fold}/{n_folds-1} | train:[0,{train_end}) val:[{train_end},{val_end}) test:[{val_end},{test_end})<<<<<<')
                exp = Exp(args)
                _, _, _, _ = exp.train(setting)
                mse, mae, rmse, mape, mspe, rse, nd, nrmse, mda, sharpe, max_dd, peak_mb, _ = exp.test(setting)
                fold_metrics.append([mse, mae, rmse, mape, mspe, rse, nd, nrmse, mda, sharpe, max_dd])
                torch.cuda.empty_cache()

            # Aggregate across folds
            fold_metrics = np.array(fold_metrics)  # (n_folds, 11)
            means = fold_metrics.mean(axis=0)
            stds  = fold_metrics.std(axis=0)
            print('\n========== Walk-forward Summary ({} folds) =========='.format(len(fold_metrics)))
            for name, m, s in zip(metric_names, means, stds):
                print('  {}: {:.6f} ± {:.6f}'.format(name, m, s))
            f = open("test_result.txt", 'a')
            f.write('\n=== Walk-forward Summary ({} folds) ===\n'.format(len(fold_metrics)))
            for name, m, s in zip(metric_names, means, stds):
                f.write('  {}: {:.6f} +/- {:.6f}\n'.format(name, m, s))
            f.write('\n')
            f.close()

        else:
            # --- Standard hold-out training ---
            args.wf_borders = None
            for ii in range(args.itr):
                setting = '{}_{}_data{}_feature{}_seql{}_labell{}_predl{}_dmodel{}_nheads{}_elayers{}_dlayers{}_dff{}_fc{}_embed{}_distil{}_{}_iter{}'.format(
                    args.model_id, args.model, args.data, args.features,
                    args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers,
                    args.d_ff, args.factor, args.embed, args.distil, args.des, ii)

                exp = Exp(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, _, _, _ = exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)  # returns 12 values; unused here

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()

        end = time.time()
        used_time = end - start
        print("time:", used_time)
        f = open("train_result.txt", 'a')
        f.write('time:{}'.format(used_time))
        f.write('\n')
        f.write('\n')
        f.close()
    else:
        ii = 0        
        setting = '{}_{}_data{}_feature{}_seql{}_labell{}_predl{}_dmodel{}_nheads{}_elayers{}_dlayers{}_dff{}_fc{}_embed{}_distil{}_{}_iter{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Dataloader that configured in data_factory.py is using num_workers==arg.num_workers
    # It generates 8 child processes to load data in parallel when num_workers=8 (default)
    # When starts running this script, the child processes are generated and starts running. However, they don't share the memory with the main process. 
    # So they will need to load the run_longExp.py again which may cause regenerating child processes again.
    # Using if __name__ == "__main__": can avoid this problem. This if __name__ == "__main__" is the guard block that tells the child processes not to run the code in this block.
    # This works for Linux and MacOS. For windows, it needs freeze_support() to support this.
    # In Windows, freeze_support() provides extra state and entry-point metadata. It installs bookkeeping. Whithout it, the spawn child process call can fail even thought the guard is present. 
    
    freeze_support() # for Windows support;
    main()
