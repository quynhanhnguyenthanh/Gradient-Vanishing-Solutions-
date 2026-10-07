"""Vẽ hình và lập bảng từ results/. Không chạy lại thí nghiệm.

    python analysis.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = 'results'
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
SEED0 = 42


def load_raw():
    """Đọc mọi file json trong results/raw."""
    out = {}
    for path in glob.glob(os.path.join(RAW_DIR, '*.json')):
        with open(path) as f:
            d = json.load(f)
        out[(d['config']['name'], d['config']['seed'])] = d
    return out


def _save(fig_name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=140)
    plt.show()


# ----------------------------------------------------------------------
# Hình 1 - ‖∇W‖/‖W‖ theo layer, thang log.  HÌNH CHÍNH.
# ----------------------------------------------------------------------
def fig1_gradient_ratio(raw, seed=SEED0, when='init'):
    key = 'init_grad_ratio' if when == 'init' else 'final_grad_ratio'
    plt.figure(figsize=(10, 5.5))

    for (name, s), d in sorted(raw.items()):
        if s != seed:
            continue
        r = d[key]
        ks = list(r.keys())
        plt.plot(ks, [r[k] for k in ks], 'o-', label=name, alpha=0.85)

    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel('Layer (input -> output)')
    plt.ylabel(r'$\|\nabla W\| / \|W\|$')
    plt.title(f'Gradient ratio theo layer ({"tại khởi tạo" if when == "init" else "cuối train"})'
              f' - seed {seed}')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, which='both')
    _save(f'fig1_gradient_ratio_{when}.png')


# ----------------------------------------------------------------------
# Hình 2 - heatmap layer x epoch
# ----------------------------------------------------------------------
def fig2_heatmap(raw, names=None, seed=SEED0):
    items = [(n, d) for (n, s), d in sorted(raw.items())
             if s == seed and (names is None or n in names)]
    if not items:
        return

    fig, axes = plt.subplots(1, len(items), figsize=(5.5 * len(items), 4.5),
                             squeeze=False)
    for ax, (name, d) in zip(axes[0], items):
        hist = d['grad_ratio_history']
        layers = list(hist[0].keys())
        mat = np.array([[ep[l] for ep in hist] for l in layers])
        im = ax.imshow(np.log10(mat + 1e-12), aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=8)
        ax.set_xlabel('Epoch')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label=r'$\log_{10}(\|\nabla W\|/\|W\|)$')
    _save('fig2_heatmap.png')


# ----------------------------------------------------------------------
# Hình 6 - accuracy theo learning rate
# Giải pháp tốt thường làm mô hình ÍT NHẠY với lr hơn: vùng lr cho
# kết quả tốt rộng hơn, không chỉ đỉnh cao hơn.
# ----------------------------------------------------------------------
def fig6_lr_sensitivity():
    path = os.path.join(RESULTS_DIR, 'lr_sweep.csv')
    if not os.path.exists(path):
        print('Chưa có lr_sweep.csv - cần sửa tune_lr để lưu.')
        return
    df = pd.read_csv(path).drop_duplicates(['name', 'lr'])

    plt.figure(figsize=(9, 5))
    for name, g in df.groupby('name'):
        g = g.sort_values('lr')
        plt.plot(g['lr'], g['val_acc'], 'o-', label=name)
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Val accuracy')
    plt.title('Độ nhạy với learning rate')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    _save('fig6_lr_sensitivity.png')


# ----------------------------------------------------------------------
# Hình 7 - ‖ΔW‖/‖W‖ theo layer, SGD vs Adam
# ----------------------------------------------------------------------
def fig7_update_ratio(raw, pair=('baseline', '+adam'), seed=SEED0):
    plt.figure(figsize=(9, 5))
    for name in pair:
        d = raw.get((name, seed))
        if d is None:
            continue
        u = d['final_update_ratio']
        ks = list(u.keys())
        plt.plot(ks, [u[k] for k in ks], 'o-', label=name)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.ylabel(r'$\|\Delta W\| / \|W\|$')
    plt.title('Mức cập nhật thực tế theo layer')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    _save('fig7_update_ratio.png')


# ----------------------------------------------------------------------
# Hình 8 - slope và R2 theo epoch
#
# TA nhắc: không kết luận "vanishing là trạng thái ổn định" chỉ từ hai
# thời điểm. Phải xem slope ở MỌI epoch, và kiểm tra R2: nếu R2 thấp thì
# suy giảm không theo cấp số nhân, slope đơn lẻ mất ý nghĩa.
# ----------------------------------------------------------------------
def fig8_slope_per_epoch(raw, seed=SEED0):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for (name, s), d in sorted(raw.items()):
        if s != seed or 'slope_per_epoch' not in d:
            continue
        ax1.plot(d['slope_per_epoch'], label=name)
        if 'r2_per_epoch' in d:
            ax2.plot(d['r2_per_epoch'], label=name)

    ax1.set_ylabel('decay_slope (bậc/layer)')
    ax1.set_title('Slope theo epoch - vanishing có ổn định không?')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axhline(0.95, ls='--', c='gray', lw=1)
    ax2.set_ylabel(r'$R^2$ của đường fit')
    ax2.set_xlabel('Epoch')
    ax2.set_title(r'$R^2$ thấp -> suy giảm KHÔNG theo cấp số nhân, '
                  'slope đơn lẻ mất ý nghĩa')
    ax2.grid(alpha=0.3)
    _save('fig8_slope_per_epoch.png')


# ----------------------------------------------------------------------
# Hình 9 - weight drift  ‖W_t − W_0‖/‖W_0‖ theo layer
#
# Đây là bằng chứng phân biệt HAI cách giải thích khi slope không đổi:
#   (a) vanishing là trạng thái ổn định  -> drift đáng kể ở mọi layer
#   (b) model gần như không học gì       -> drift ~ 0 ở các layer đầu
# ----------------------------------------------------------------------
def fig9_weight_drift(raw, seed=SEED0):
    plt.figure(figsize=(9, 5))
    for (name, s), d in sorted(raw.items()):
        if s != seed or 'drift_final' not in d:
            continue
        dr = d['drift_final']
        ks = list(dr.keys())
        plt.plot(ks, [dr[k] for k in ks], 'o-', label=name)

    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.ylabel(r'$\|W_t - W_0\| / \|W_0\|$')
    plt.title('Trọng số từng layer đã dịch chuyển bao xa khỏi điểm khởi tạo')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, which='both')
    _save('fig9_weight_drift.png')

    print('\nĐọc hình 9: drift ~ 0 ở các layer đầu nghĩa là layer đó gần như')
    print('không học. Khi đó slope không đổi KHÔNG phải bằng chứng cho')
    print('"vanishing ổn định", mà chỉ phản ánh model không học được gì.')


# ----------------------------------------------------------------------
# Hình 10 - sparsity vs dead-ReLU rate
#
# TA phân biệt rõ hai chỉ số:
#   sparsity      = tỉ lệ output bằng 0 TRÊN MỖI MẪU (tính thưa, có ích)
#   dead-ReLU     = tỉ lệ neuron bằng 0 với MỌI MẪU  (neuron chết thật)
# Chỉ đo sparsity sẽ không phân biệt được hai thứ này.
# ----------------------------------------------------------------------
def fig10_sparsity_vs_dead(raw, names=None, seed=SEED0):
    items = [(n, d) for (n, s), d in sorted(raw.items())
             if s == seed and (names is None or n in names)
             and 'init_sparsity' in d]
    if not items:
        print('Chưa có init_sparsity - cần sửa measure_at_init để lưu.')
        return

    plt.figure(figsize=(9, 5))
    for name, d in items:
        ks = list(d['init_sparsity'].keys())
        plt.plot(ks, [d['init_sparsity'][k] for k in ks], 'o-',
                 label=f'{name} sparsity')
        if 'init_dead_relu' in d:
            plt.plot(ks, [d['init_dead_relu'][k] for k in ks], 's--',
                     label=f'{name} dead-ReLU')

    plt.xticks(rotation=45)
    plt.ylabel('Tỉ lệ')
    plt.title('Sparsity vs Dead-ReLU rate theo layer')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _save('fig10_sparsity_vs_dead.png')


# ----------------------------------------------------------------------
# Bảng đối chứng Adam
#
# TA nhắc: không kết luận từ MỘT chỉ số. Giả thuyết "Adam bù mức cập nhật"
# chỉ được ủng hộ khi ĐỒNG THỜI thấy:
#   (a) gradient tại layer đầu của Adam KHÔNG khá hơn SGD đáng kể
#   (b) decay_slope hai bên tương đương
#   (c) update_ratio tại layer đầu của Adam LỚN HƠN rõ rệt
#   (d) accuracy của Adam cao hơn
# ----------------------------------------------------------------------
def adam_table(raw, pair=('baseline', '+adam')):
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'summary.csv'))
    df = df[df['name'].isin(pair)]

    cols = ['init_grad_ratio_L1', 'final_grad_ratio_L1', 'init_decay_slope',
            'final_update_ratio_L1', 'final_update_ratio_out', 'test_acc']
    cols = [c for c in cols if c in df.columns]
    agg = df.groupby('name')[cols].agg(['mean', 'std']).round(5)

    print('\n=== Đối chứng Adam vs SGD (cùng kiến trúc, cùng init) ===')
    print(agg.to_string())

    print('\nĐọc bảng:')
    print('  (a)(b) gradient và slope hai bên tương đương  -> Adam không khôi phục dòng gradient')
    print('  (c)    update_ratio layer đầu của Adam lớn hơn -> Adam bù bằng bước cập nhật')
    print('  (d)    accuracy Adam cao hơn')
    print('  Đủ cả 4 mới ủng hộ giả thuyết. Thiếu bất kỳ điều nào -> KHÔNG kết luận.')
    return agg


# ----------------------------------------------------------------------
def main_table():
    path = os.path.join(RESULTS_DIR, 'summary.csv')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print('Chưa có summary.csv.')
        return None
    df = pd.read_csv(path)
    cols = {'test_acc': ['mean', 'std'],
            'init_decay_slope': 'mean',
            'final_r2': 'mean',
            'drift_L1': 'mean',
            'drift_out': 'mean',
            'stopped_at': 'mean'}
    cols = {k: v for k, v in cols.items() if k in df.columns}
    for t in ['0.001', '0.01', '0.05']:
        c = f'init_eff_depth_{t}'
        if c in df.columns:
            cols[c] = 'mean'

    agg = df.groupby('name').agg(cols).round(4)
    print('\n=== Bảng kết quả Pha 1 ===')
    print(agg.to_string())
    print('\nChỉ kết luận A tốt hơn B khi chênh lệch LỚN HƠN RÕ RỆT so với std.')
    print('Nếu nằm trong khoảng std -> ghi "chưa phân biệt được".')
    print('effective_depth ở 3 ngưỡng: nếu thứ hạng KHÔNG đổi qua cả ba,')
    print('kết luận vững; nếu đổi, đó là phát hiện phải nêu.')
    return agg


if __name__ == '__main__':
    raw = load_raw()
    print(f'Đọc {len(raw)} run từ {RAW_DIR}')

    main_table()
    fig1_gradient_ratio(raw, when='init')
    fig1_gradient_ratio(raw, when='final')
    fig2_heatmap(raw, names=['baseline', '+relu', '+residual'])
    fig6_lr_sensitivity()
    fig7_update_ratio(raw)
    fig8_slope_per_epoch(raw)
    fig9_weight_drift(raw)
    fig10_sparsity_vs_dead(raw, names=['baseline', '+relu'])
    adam_table(raw)
