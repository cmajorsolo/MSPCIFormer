import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def NRMSE(pred, true):
    return np.sqrt(np.mean(np.power((pred - true), 2))) / (np.mean(np.abs(true)))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def MDA(pred, true):
    # Use only first predicted step: shape (n_samples, pred_len, n_features) -> (n_samples,)
    pred_first = pred[:, 0, -1]  # first pred step, target feature
    true_first = true[:, 0, -1]  # first true step
    true_prev = np.concatenate([[true_first[0]], true_first[:-1]])  # true[t-1]
    pred_dir = np.sign(pred_first - true_prev)
    true_dir = np.sign(true_first - true_prev)
    return np.mean(pred_dir == true_dir)


def strategy_returns(pred_inv, true_inv):
    # pred_inv, true_inv: inverse-transformed prices, shape (n_samples, pred_len, n_features)
    pred_first = pred_inv[:, 0, -1]
    true_first = true_inv[:, 0, -1]
    true_prev = np.concatenate([[true_first[0]], true_first[:-1]])  # true[t-1] as entry price
    signal = np.sign(pred_first - true_prev)  # +1 long, -1 short
    actual_return = (true_first - true_prev) / (np.abs(true_prev) + 1e-12)
    return signal * actual_return


def sharpe_ratio(returns, annualization=365):
    if np.std(returns) < 1e-12:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(annualization)


def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    running_peak = np.maximum.accumulate(cumulative)
    drawdown = (running_peak - cumulative) / (running_peak + 1e-12)
    return np.max(drawdown)


def finance_metric(pred_inv, true_inv, annualization=365):
    mda = MDA(pred_inv, true_inv)
    returns = strategy_returns(pred_inv, true_inv)
    sharpe = sharpe_ratio(returns, annualization)
    max_dd = max_drawdown(returns)
    return mda, sharpe, max_dd


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    nd = ND(pred,true)
    nrmse = NRMSE(pred,true)

    return mae, mse, rmse, mape, mspe, rse , corr, nd, nrmse

def metric2(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    nd = ND(pred,true)
    nrmse = NRMSE(pred,true)

    return mae, mse, rmse, mape, mspe, rse , nd, nrmse