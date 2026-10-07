"""Nạp dữ liệu lên GPU + chạy một thí nghiệm."""

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import build_mlp
from metrics import (grad_ratio_per_layer, grad_abs_mean_per_layer,
                     decay_slope, decay_slope_with_r2, weight_drift,
                     relative_gradient, effective_depth,
                     snapshot_weights, update_ratio_per_layer,
                     ActivationRecorder, saturation_rate,
                     sparsity_rate, dead_relu_rate)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_CACHE = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(train_ratio=0.9, split_seed=42):
    """Nạp FashionMNIST lên GPU một lần, cache lại cho mọi run."""
    if 'data' in _CACHE:
        return _CACHE['data']

    tr = FashionMNIST('./data', train=True, download=True,
                      transform=transforms.ToTensor())
    te = FashionMNIST('./data', train=False, download=True,
                      transform=transforms.ToTensor())

    X = tr.data.float().div(255.).view(-1, 784)   # BẮT BUỘC chia 255
    y = tr.targets
    Xte = te.data.float().div(255.).view(-1, 784)
    yte = te.targets

    g = torch.Generator().manual_seed(split_seed)  # split cố định mọi run
    perm = torch.randperm(len(X), generator=g)
    n_tr = int(len(X) * train_ratio)
    a, b = perm[:n_tr], perm[n_tr:]

    data = (X[a].to(device), y[a].to(device),
            X[b].to(device), y[b].to(device),
            Xte.to(device), yte.to(device))
    _CACHE['data'] = data
    return data


@torch.no_grad()
def evaluate(model, X, y, criterion, bs=2048):
    model.eval()
    loss_sum, correct, steps = 0.0, 0, 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += criterion(out, y[i:i + bs]).item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
        steps += 1
    return loss_sum / steps, correct / len(X)


def measure_at_init(model, X, y, criterion, bs):
    """Đo trước bước cập nhật đầu tiên. KHÔNG phụ thuộc learning rate."""
    model.train()
    rec = ActivationRecorder(model)
    criterion(model(X[:bs]), y[:bs]).backward()

    ratios = grad_ratio_per_layer(model)
    result = {
        'grad_ratio': ratios,
        'grad_abs': grad_abs_mean_per_layer(model),
        'decay_slope': decay_slope(ratios, model),
        'relative_gradient': relative_gradient(ratios, model),
        'effective_depth': effective_depth(ratios),
        'saturation': saturation_rate(rec.store),
        'sparsity': sparsity_rate(rec.store),
        'dead_relu': dead_relu_rate(rec.store),
    }
    model.zero_grad()
    rec.remove()
    return result


def run_experiment(cfg, verbose=True):
    set_seed(cfg.seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()

    model = build_mlp(cfg).to(device)
    w0 = {n: l.weight.detach().clone() for n, l in model.weight_layers()}
    criterion = nn.CrossEntropyLoss()
    opt = (optim.Adam if cfg.optimizer == 'adam' else optim.SGD)(
        model.parameters(), lr=cfg.lr)

    hist = {'train_loss': [], 'val_loss': [], 'val_acc': [],
            'grad_ratio': [], 'update_ratio': [], 'slope': [], 'r2': [], 'drift': []}
    init = measure_at_init(model, X_tr, y_tr, criterion, cfg.batch_size)

    n = len(X_tr)
    best_val, wait, stopped_at = float('inf'), 0, cfg.epochs
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum, steps = 0.0, 0
        ep_grad, ep_update = None, None

        for step, i in enumerate(range(0, n, cfg.batch_size)):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()

            if step == 0:                       # chỉ đo batch đầu mỗi epoch
                ep_grad = grad_ratio_per_layer(model)
                snap = snapshot_weights(model)
                opt.step()
                ep_update = update_ratio_per_layer(model, snap)
            else:
                opt.step()

            loss_sum += loss.item()
            steps += 1

        val_loss, val_acc = evaluate(model, X_va, y_va, criterion)
        hist['train_loss'].append(loss_sum / steps)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)
        hist['grad_ratio'].append(ep_grad)
        hist['update_ratio'].append(ep_update)

        s, r2 = decay_slope_with_r2(ep_grad, model)
        hist['slope'].append(s)
        hist['r2'].append(r2)
        hist['drift'].append(weight_drift(model, w0))

        if val_loss < best_val - cfg.min_delta:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                stopped_at = epoch + 1
                break

    _, test_acc = evaluate(model, X_te, y_te, criterion)
    final_ratio = hist['grad_ratio'][-1]

    result = {
        **cfg.to_dict(),
        'test_acc': test_acc,
        'val_acc': hist['val_acc'][-1],
        'final_train_loss': hist['train_loss'][-1],
        'stopped_at': stopped_at,
        'runtime_s': time.time() - t0,
        'init_decay_slope': init['decay_slope'],
        'final_decay_slope': decay_slope(final_ratio, model),
        'init_grad_ratio_L1': init['grad_ratio']['layer1'],
        'final_grad_ratio_L1': final_ratio['layer1'],
        'final_update_ratio_L1': hist['update_ratio'][-1]['layer1'],
        'final_update_ratio_out': hist['update_ratio'][-1]['output'],
        'final_r2': hist['r2'][-1],
        'drift_L1': hist['drift'][-1]['layer1'],
        'drift_out': hist['drift'][-1]['output'],
        'slope_change': hist['slope'][-1] - hist['slope'][0],
    }
    for t, v in init['effective_depth'].items():
        result[f'init_eff_depth_{t}'] = v
    for t, v in effective_depth(final_ratio).items():
        result[f'final_eff_depth_{t}'] = v

    if verbose:
        print(f"[{cfg.name} | seed {cfg.seed}] "
              f"test_acc {test_acc:.4f} | loss {result['final_train_loss']:.4f} | "
              f"slope {result['final_decay_slope']:.3f} | "
              f"stop@{stopped_at} | {result['runtime_s']:.1f}s")

    return result, hist, init

