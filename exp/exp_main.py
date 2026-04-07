from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import (
    Informer,
    Autoformer,
    DLinear,
    MSGNet,
    Transformer,
    PatchTST,
    iTransformer,
    MSPCIFormer,
    TimeXer,
    TimesNet,
    NBeats,
    LSTMModel,
)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, finance_metric

import torch
import torch.nn as nn
from torch import optim, autograd
from torch.optim import lr_scheduler

import os
import time
import psutil

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        # Adjust seq_len for RAG datasets BEFORE parent init (which builds model)
        if args.data == 'rag':
            self.original_seq_len = args.seq_len
            args.seq_len = args.k_retrieval * args.chunk_size + args.seq_len
            print(f"RAG: Adjusted model seq_len from {self.original_seq_len} to {args.seq_len}")

        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'MSGNet': MSGNet,
            'Transformer': Transformer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'MSPCIFormer': MSPCIFormer,
            'TimeXer': TimeXer,
            'TimesNet': TimesNet,
            'NBeats': NBeats,
            'LSTM': LSTMModel,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    #flag = 'train' or 'val' or 'test'
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                # Handle both 4-tuple and 5-tuple (MSPFormerCross with RAG)
                if len(batch_data) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, rag_chunks = batch_data
                    rag_chunks = rag_chunks.float().to(self.device) if rag_chunks is not None else None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    rag_chunks = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():             
                        if ('Linear' in self.args.model or 'TST' in self.args.model or 'MSPCIFormer' in self.args.model or self.args.model == 'NBeats'):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:                                       
                    if ('Linear' in self.args.model or 'TST' in self.args.model or 'MSPCIFormer' in self.args.model or self.args.model == 'NBeats'):
                        outputs = self.model(batch_x)                    
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_vali_loss = np.inf
        epoch_times = []  # track wall-clock time per epoch

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        #use automatic mixed precision training
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                # Handle both 4-tuple and 5-tuple (MSPFormerCross with RAG)
                if len(batch_data) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, rag_chunks = batch_data
                    rag_chunks = rag_chunks.float().to(self.device) if rag_chunks is not None else None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    rag_chunks = None

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():         
                        if ('Linear' in self.args.model or 'TST' in self.args.model or 'MSPCIFormer' in self.args.model or self.args.model == 'NBeats'):
                            outputs = self.model(batch_x)   
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if ('Linear' in self.args.model or "TST" in self.args.model or "MSPCIFormer" in self.args.model or self.args.model == 'NBeats'):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention: #whether to output attention in ecoder
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    with autograd.detect_anomaly():
                        loss.backward()
                        model_optim.step()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            elapsed = time.time() - epoch_time
            epoch_times.append(elapsed)
            print("Epoch: {} cost time: {}".format(epoch + 1, elapsed))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            f = open("train_result.txt", 'a')
            f.write(setting + "  \n")
            f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            f.write('\n')
            f.close()

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))   

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        avg_time_per_epoch = float(np.mean(epoch_times)) if epoch_times else 0.0
        return self.model, best_vali_loss, avg_time_per_epoch, epoch_times

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Memory tracking: reset GPU peak counter; record CPU baseline
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _process = psutil.Process()
        _mem_baseline_mb = _process.memory_info().rss / 1024 / 1024

        # Inference latency tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _infer_start = time.time()
        _n_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                # Handle both 4-tuple and 5-tuple (MSPFormerCross with RAG)
                if len(batch_data) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, rag_chunks = batch_data
                    rag_chunks = rag_chunks.float().to(self.device) if rag_chunks is not None else None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    rag_chunks = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['MSPFormerCross', 'MSPFormerCrossV2', 'MSPFormerCrossV3', 'MSPFormerCrossV4']:
                            outputs = self.model(batch_x, rag_chunks=rag_chunks)
                        elif ('Linear' in self.args.model or "TST" in self.args.model or "MSPCIFormer" in self.args.model or self.args.model == 'NBeats'):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model in ['MSPFormerCross', 'MSPFormerCrossV2', 'MSPFormerCrossV3', 'MSPFormerCrossV4']:
                        outputs = self.model(batch_x, rag_chunks=rag_chunks)
                    elif ('Linear' in self.args.model or "TST" in self.args.model or "MSPCIFormer" in self.args.model or self.args.model == 'NBeats'):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                _n_samples += pred.shape[0]
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        #See utils / tools for usage
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        # print('preds_shape:', len(preds),len(preds[0]),len(preds[1]))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        print('preds_shape:', preds.shape)
        print('trues_shape:', trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, nd, nrmse = metric(preds, trues)
        print('nd:{}, nrmse:{}, mse:{}, rmse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse, mse, rmse, mae, rse, mape))

        # Inverse-transform for finance metrics (real price scale)
        # preds/trues have shape (n_samples, pred_len, n_features) where n_features=1 for MS task.
        # The scaler was fitted on all columns; pad to full width, inverse-transform, then extract target (last col).
        n_samples, pred_len, n_features = preds.shape
        n_scaler_features = test_data.scaler.n_features_in_
        def _inverse(arr):
            flat = arr.reshape(-1, n_features)  # (n_samples*pred_len, n_features)
            if n_features < n_scaler_features:
                pad = np.zeros((flat.shape[0], n_scaler_features - n_features))
                flat_padded = np.concatenate([pad, flat], axis=1)  # target is last col
            else:
                flat_padded = flat
            inv = test_data.inverse_transform(flat_padded)
            return inv[:, -n_features:].reshape(n_samples, pred_len, n_features)
        preds_inv = _inverse(preds)
        trues_inv = _inverse(trues)
        mda, sharpe, max_dd = finance_metric(preds_inv, trues_inv, annualization=365)
        print('mda:{}, sharpe:{}, max_drawdown:{}'.format(mda, sharpe, max_dd))

        # Peak memory used during inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory_mb = _process.memory_info().rss / 1024 / 1024 - _mem_baseline_mb

        # Inference latency
        _infer_total_s = time.time() - _infer_start
        latency_ms_per_sample = (_infer_total_s / _n_samples * 1000) if _n_samples > 0 else 0.0
        print('peak_memory_mb:{:.2f}  infer_total_s:{:.4f}  latency_ms/sample:{:.4f}'.format(
            peak_memory_mb, _infer_total_s, latency_ms_per_sample))

        f = open("test_result.txt", 'a')
        f.write(setting + "  \n")
        f.write('nd:{}, nrmse:{}, mse:{}, rmse:{}, mae:{}, rse:{}, mape:{}'.format(nd, nrmse, mse, rmse, mae, rse, mape))
        f.write('\n')
        f.write('mda:{}, sharpe:{}, max_drawdown:{}'.format(mda, sharpe, max_dd))
        f.write('\n')
        f.write('peak_memory_mb:{:.2f}, infer_total_s:{:.4f}, latency_ms_per_sample:{:.4f}'.format(
            peak_memory_mb, _infer_total_s, latency_ms_per_sample))
        f.write('\n')
        f.write('\n')
        f.close()

        # Ensure all metrics are scalars
        metrics_array = np.array([
            float(mae), float(mse), float(rmse), float(mape),
            float(mspe), float(rse), float(np.mean(corr)),  # Force corr to be scalar
            float(mda), float(sharpe), float(max_dd)
        ])
        np.save(folder_path + 'metrics.npy', metrics_array)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return mse, mae, rmse, mape, mspe, rse, nd, nrmse, mda, sharpe, max_dd, peak_memory_mb, latency_ms_per_sample


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(pred_loader):
                # Handle both 4-tuple and 5-tuple (MSPFormerCross with RAG)
                if len(batch_data) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, rag_chunks = batch_data
                    rag_chunks = rag_chunks.float().to(self.device) if rag_chunks is not None else None
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    rag_chunks = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
                batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in ['MSPFormerCross', 'MSPFormerCrossV2', 'MSPFormerCrossV3', 'MSPFormerCrossV4']:
                            outputs = self.model(batch_x, rag_chunks=rag_chunks)
                        elif ('Linear' in self.args.model or "TST" in self.args.model or "MSCIPFormer" in self.args.model):
                            outputs = self.model(batch_x)                       
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model in ['MSPFormerCross', 'MSPFormerCrossV2', 'MSPFormerCrossV3', 'MSPFormerCrossV4']:
                        outputs = self.model(batch_x, rag_chunks=rag_chunks)
                    elif ('Linear' in self.args.model or "TST" in self.args.model or "MSPCIFormer" in self.args.model or self.args.model == 'NBeats'):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
