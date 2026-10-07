"""Chạy toàn bộ thí nghiệm.

    python run_all.py --phase 1
    python run_all.py --phase 2
    python run_all.py --phase 3
    python run_all.py --phase all

Ghi tăng dần vào results/summary.csv nên mất kết nối Colab không mất kết quả:
chạy lại sẽ bỏ qua các run đã xong.
"""

import argparse
import json
import os
from dataclasses import replace

import pandas as pd

from config import Config, SEEDS, LR_GRID
from train import run_experiment

RESULTS_DIR = 'results'
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
SUMMARY = os.path.join(RESULTS_DIR, 'summary.csv')
LR_SWEEP = os.path.join(RESULTS_DIR, 'lr_sweep.csv')


# ----------------------------------------------------------------------
# Định nghĩa cấu hình từng pha
# ----------------------------------------------------------------------
BASELINE = Config(name='baseline')

# 5 giải pháp, mỗi cái đại diện một cơ chế khác nhau (theo góp ý TA)
SOLUTIONS = {
    'relu':     dict(activation='relu'),
    'he':       dict(init='he'),
    'bn':       dict(norm='batchnorm'),
    'residual': dict(residual=True),
    'adam':     dict(optimizer='adam'),
}

# Cấu hình đầy đủ, dùng làm gốc cho Pha 3
FULL = Config(name='full', activation='relu', init='he',
              norm='batchnorm', residual=True, optimizer='adam')


def phase1_configs():
    """Cô lập từng biến: baseline + 5 giải pháp, mỗi cái đổi ĐÚNG một thứ."""
    out = [replace(BASELINE, name='baseline')]
    for key, kw in SOLUTIONS.items():
        out.append(replace(BASELINE, name=f'+{key}', **kw))
    return out


def phase2_configs(depths=(3, 7, 15, 25)):
    """Khảo sát theo độ sâu. Vanishing là hiện tượng theo độ sâu nên đây là
    pha gắn chặt nhất với câu hỏi nghiên cứu. TA cũng chỉ ra chênh lệch slope
    nhỏ tạo khác biệt lớn khi nhân qua nhiều layer."""
    out = []
    for cfg in phase1_configs():
        for d in depths:
            out.append(replace(cfg, name=f'{cfg.name}_d{d}', depth=d))
    return out


def phase3_configs():
    """Leave-one-out: bỏ từng giải pháp khỏi cấu hình đầy đủ.

    Dùng leave-one-out thay vì cộng dồn vì kết quả cộng dồn phụ thuộc
    thứ tự thêm - giải pháp thêm sau luôn có vẻ đóng góp ít hơn.
    """
    removal = {
        'relu':     dict(activation='sigmoid'),
        'he':       dict(init='normal_0.05'),
        'bn':       dict(norm='none'),
        'residual': dict(residual=False),
        'adam':     dict(optimizer='sgd'),
    }
    out = [replace(FULL, name='full')]
    for key, kw in removal.items():
        out.append(replace(FULL, name=f'full-{key}', **kw))
    return out


# ----------------------------------------------------------------------
# Tune learning rate: cùng ngân sách cho mọi cấu hình
# ----------------------------------------------------------------------
def tune_lr(cfg, tune_epochs=20):
    """Thử cùng một lưới 4 giá trị, chọn theo val_acc, dùng 1 seed.

    Không cố định lr chung vì thang lr của SGD và Adam khác hẳn nhau.
    Không tune tùy ý vì cấu hình được tune kỹ hơn sẽ có lợi thế.
    """
    grid = LR_GRID[cfg.optimizer]
    best_lr, best_acc, sweep = grid[0], -1.0, []

    for lr in grid:
        trial = replace(cfg, lr=lr, epochs=tune_epochs,
                        patience=tune_epochs, seed=SEEDS[0])
        res, _, _ = run_experiment(trial, verbose=False)
        print(f"    lr={lr:<8g} val_acc={res['val_acc']:.4f}")
        sweep.append({'name': cfg.name, 'lr': lr, 'val_acc': res['val_acc']})
        if res['val_acc'] > best_acc:
            best_lr, best_acc = lr, res['val_acc']

    os.makedirs(RESULTS_DIR, exist_ok=True)
    header = (not os.path.exists(LR_SWEEP)) or os.path.getsize(LR_SWEEP) == 0
    pd.DataFrame(sweep).to_csv(LR_SWEEP, mode='a', header=header, index=False)

    print(f"    -> chọn lr={best_lr:g}")
    return best_lr


# ----------------------------------------------------------------------
# Lưu kết quả
# ----------------------------------------------------------------------
def load_done():
    """Trả tập (name, seed) đã chạy xong, để resume.

    Phòng ba trường hợp: file chưa có, file rỗng, file thiếu cột.
    """
    if not os.path.exists(SUMMARY) or os.path.getsize(SUMMARY) == 0:
        return set()
    try:
        df = pd.read_csv(SUMMARY)
    except Exception:
        return set()
    if 'name' not in df.columns or 'seed' not in df.columns:
        return set()
    return set(zip(df['name'], df['seed']))


def append_row(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    header = (not os.path.exists(SUMMARY)) or os.path.getsize(SUMMARY) == 0
    pd.DataFrame([row]).to_csv(SUMMARY, mode='a', header=header, index=False)


def save_raw(cfg, hist, init):
    """Lưu đường gradient theo layer và theo epoch để vẽ hình sau,
    không phải chạy lại thí nghiệm."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f'{cfg.name}_s{cfg.seed}.json')
    data = {
        'config': cfg.to_dict(),
        'init_grad_ratio': init['grad_ratio'],
        'init_saturation': init.get('saturation'),
        'init_sparsity': init.get('sparsity'),
        'init_dead_relu': init.get('dead_relu'),
        'init_relative_gradient': init.get('relative_gradient'),
        'final_grad_ratio': hist['grad_ratio'][-1],
        'final_update_ratio': hist['update_ratio'][-1],
        'val_acc_curve': hist['val_acc'],
        'train_loss_curve': hist['train_loss'],
    }
    # Ba khóa dưới đây trả lời phản biện của TA về slope
    if hist.get('slope'):
        data['slope_per_epoch'] = hist['slope']
    if hist.get('r2'):
        data['r2_per_epoch'] = hist['r2']
    if hist.get('drift'):
        data['drift_final'] = hist['drift'][-1]

    with open(path, 'w') as f:
        json.dump(data, f)


# ----------------------------------------------------------------------
def run_phase(configs, phase_name, tune=True):
    done = load_done()
    print(f"\n{'=' * 60}\nPHA {phase_name}: {len(configs)} cấu hình "
          f"x {len(SEEDS)} seed\n{'=' * 60}")

    for cfg in configs:
        if all((cfg.name, s) in done for s in SEEDS):
            print(f"[bỏ qua] {cfg.name} - đã chạy xong")
            continue

        print(f"\n--- {cfg.name} ---")
        lr = tune_lr(cfg) if tune else cfg.lr

        for seed in SEEDS:                      # cùng bộ seed cho MỌI cấu hình
            if (cfg.name, seed) in done:
                continue
            run_cfg = replace(cfg, lr=lr, seed=seed)
            res, hist, init = run_experiment(run_cfg)
            res['phase'] = phase_name
            append_row(res)
            save_raw(run_cfg, hist, init)


def summarize():
    if not os.path.exists(SUMMARY) or os.path.getsize(SUMMARY) == 0:
        print('Chưa có kết quả.')
        return
    try:
        df = pd.read_csv(SUMMARY)
    except Exception as e:
        print(f'Không đọc được {SUMMARY}: {e}')
        return
    if 'name' not in df.columns:
        print(f'summary.csv thiếu cột "name". Cột hiện có: {list(df.columns)}')
        print('File có thể bị hỏng - xóa results/ và chạy lại.')
        return

    wanted = {
        'test_acc': ['mean', 'std'],
        'init_decay_slope': 'mean',
        'final_r2': 'mean',
        'drift_L1': 'mean',
        'drift_out': 'mean',
        'init_eff_depth_0.01': 'mean',
        'seed': 'count',
    }
    wanted = {k: v for k, v in wanted.items() if k in df.columns}
    agg = df.groupby('name').agg(wanted).round(5)

    print('\n' + agg.to_string())
    print('\nLưu ý: chỉ kết luận cấu hình A tốt hơn B khi chênh lệch lớn hơn '
          'rõ rệt so với std.')
    print('Nếu nằm trong khoảng std -> ghi "chưa phân biệt được".')
    if 'drift_L1' in df.columns:
        print('drift_L1 ~ 0 nghĩa là layer đầu gần như không học. Khi đó slope '
              'không đổi KHÔNG phải bằng chứng cho "vanishing ổn định".')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='1', choices=['1', '2', '3', 'all'])
    ap.add_argument('--no-tune', action='store_true',
                    help='bỏ qua tune lr, dùng lr trong Config')
    args = ap.parse_args()
    tune = not args.no_tune

    if args.phase in ('1', 'all'):
        run_phase(phase1_configs(), '1_ablation', tune)
    if args.phase in ('2', 'all'):
        run_phase(phase2_configs(), '2_depth', tune)
    if args.phase in ('3', 'all'):
        run_phase(phase3_configs(), '3_leave_one_out', tune)

    summarize()
