"""Nạp dữ liệu lên GPU + chạy một thí nghiệm.

Khác tutorial:
  - toàn bộ dataset nằm sẵn trên GPU, không dùng DataLoader
  - batch_size 256 (tutorial không nêu giá trị)
  - 50 epoch + early stopping thay vì 100 epoch cố định
  - đo CẢ activation gradient và weight gradient (theo hướng dẫn TA)
"""

import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import build_mlp
from metrics import (
    # weight gradient
    grad_ratio_per_layer, grad_rms_per_layer, grad_abs_mean_per_layer,
    effective_update_ratio,
    # activation gradient
    act_grad_rms, act_grad_relative,
    # suy giảm theo độ sâu
    decay_slope, decay_slope_with_r2, relative_gradient, effective_depth,
    # mức cập nhật
    snapshot_weights, update_ratio_per_layer, weight_drift,
    # nguyên nhân
    ActivationRecorder, saturation_rate, sparsity_rate, dead_relu_rate,
    act_norm_per_layer,
)

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
    """Cộng dồn dưới dạng tensor, chỉ .item() một lần ở cuối
    để tránh đồng bộ GPU-CPU mỗi batch."""
    model.eval()
    loss_sum = torch.zeros((), device=X.device)
    correct = torch.zeros((), device=X.device)
    steps = 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        loss_sum += criterion(out, y[i:i + bs])
        correct += (out.argmax(1) == y[i:i + bs]).sum()
        steps += 1
    return (loss_sum / steps).item(), (correct / len(X)).item()


def measure_at_init(model, X, y, criterion, bs, lr):
    """Đo trước bước cập nhật đầu tiên. KHÔNG phụ thuộc learning rate
    (trừ eff_update, vốn nhân lr vào một cách tường minh)."""
    model.train()
    rec = ActivationRecorder(model, capture_grad=True)
    criterion(model(X[:bs]), y[:bs]).backward()

    ratios = grad_ratio_per_layer(model)
    result = {
        # --- cơ chế: activation gradient ---
        'act_grad_rms': act_grad_rms(rec.grads),
        'act_grad_relative': act_grad_relative(rec.grads),
        # --- hệ quả: weight gradient ---
        'grad_ratio': ratios,
        'grad_rms': grad_rms_per_layer(model),
        'grad_abs': grad_abs_mean_per_layer(model),
        'eff_update': effective_update_ratio(model, lr),
        # --- suy giảm theo độ sâu ---
        'decay_slope': decay_slope(ratios, model),
        'relative_gradient': relative_gradient(ratios, model),
        'effective_depth': effective_depth(ratios),
        # --- nguyên nhân ---
        'saturation': saturation_rate(rec.store),
        'sparsity': sparsity_rate(rec.store),
        'dead_relu': dead_relu_rate(rec.store),
        'act_norm': act_norm_per_layer(rec.store),
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
            'grad_ratio': [], 'act_grad': [], 'update_ratio': [],
            'slope': [], 'r2': [], 'drift': [], 'epoch_measured': []}

    init = measure_at_init(model, X_tr, y_tr, criterion,
                           cfg.batch_size, cfg.lr)

    n = len(X_tr)
    best_val, wait, stopped_at = float('inf'), 0, cfg.epochs
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        ep_grad = ep_update = ep_act = None

        # Chỉ đo ở epoch đầu, epoch cuối, và mỗi measure_every epoch.
        # Mỗi lần đo tốn ~40 lần đồng bộ GPU-CPU nên đo mỗi epoch rất chậm.
        do_measure = (epoch % cfg.measure_every == 0
                      or epoch == cfg.epochs - 1)

        for step, i in enumerate(range(0, n, cfg.batch_size)):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()

            if step == 0 and do_measure:
                rec = ActivationRecorder(model, capture_grad=True)
                out = model(X_tr[idx])
                loss = criterion(out, y_tr[idx])
                loss.backward()
                ep_grad = grad_ratio_per_layer(model)
                ep_act = act_grad_rms(rec.grads)
                rec.remove()
                snap = snapshot_weights(model)
                opt.step()
                ep_update = update_ratio_per_layer(model, snap)
            else:
                out = model(X_tr[idx])
                loss = criterion(out, y_tr[idx])
                loss.backward()
                opt.step()

            loss_sum += loss.detach()
            n_steps += 1

        val_loss, val_acc = evaluate(model, X_va, y_va, criterion)
        hist['train_loss'].append((loss_sum / n_steps).item())
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)
        hist['epoch_measured'].append(epoch if ep_grad is not None else None)

        if ep_grad is not None:
            hist['grad_ratio'].append(ep_grad)
            hist['act_grad'].append(ep_act)
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
                if ep_grad is None:      # epoch này chưa đo -> đo bù
                    opt.zero_grad()
                    rec = ActivationRecorder(model, capture_grad=True)
                    criterion(model(X_tr[:cfg.batch_size]),
                              y_tr[:cfg.batch_size]).backward()
                    ep_grad = grad_ratio_per_layer(model)
                    hist['grad_ratio'].append(ep_grad)
                    hist['act_grad'].append(act_grad_rms(rec.grads))
                    hist['update_ratio'].append(
                        hist['update_ratio'][-1] if hist['update_ratio'] else {})
                    s, r2 = decay_slope_with_r2(ep_grad, model)
                    hist['slope'].append(s)
                    hist['r2'].append(r2)
                    hist['drift'].append(weight_drift(model, w0))
                    rec.remove()
                    model.zero_grad()
                break

    _, test_acc = evaluate(model, X_te, y_te, criterion)
    final_ratio = hist['grad_ratio'][-1]
    last_layer = f'layer{cfg.depth}'

    result = {
        **cfg.to_dict(),
        'test_acc': test_acc,
        'val_acc': hist['val_acc'][-1],
        'final_train_loss': hist['train_loss'][-1],
        'stopped_at': stopped_at,
        'runtime_s': time.time() - t0,
        # weight gradient
        'init_decay_slope': init['decay_slope'],
        'final_decay_slope': decay_slope(final_ratio, model),
        'final_r2': hist['r2'][-1],
        'slope_change': hist['slope'][-1] - hist['slope'][0],
        'init_grad_ratio_L1': init['grad_ratio'].get('layer1'),
        'final_grad_ratio_L1': final_ratio.get('layer1'),
        # activation gradient - CƠ CHẾ
        'init_act_grad_L1': init['act_grad_rms'].get('layer1'),
        'init_act_grad_last': init['act_grad_rms'].get(last_layer),
        'final_act_grad_L1': (hist['act_grad'][-1] or {}).get('layer1'),
        # mức cập nhật
        'init_eff_update_L1': init['eff_update'].get('layer1'),
        'final_update_ratio_L1': hist['update_ratio'][-1].get('layer1'),
        'final_update_ratio_out': hist['update_ratio'][-1].get('output'),
        'drift_L1': hist['drift'][-1].get('layer1'),
        'drift_out': hist['drift'][-1].get('output'),
        # nguyên nhân
        'init_saturation_L1': init['saturation'].get('layer1'),
        'init_dead_relu_L1': init['dead_relu'].get('layer1'),
        'init_sparsity_L1': init['sparsity'].get('layer1'),
    }
    for t, v in init['effective_depth'].items():
        result[f'init_eff_depth_{t}'] = v
    for t, v in effective_depth(final_ratio).items():
        result[f'final_eff_depth_{t}'] = v

    if verbose:
        print(f"[{cfg.name} | seed {cfg.seed}] "
              f"acc {test_acc:.4f} | loss {result['final_train_loss']:.4f} | "
              f"slope {result['final_decay_slope']:.3f} "
              f"(R2 {result['final_r2']:.3f}) | "
              f"driftL1 {result['drift_L1']:.2e} | "
              f"stop@{stopped_at} | {result['runtime_s']:.1f}s")

    return result, hist, init
