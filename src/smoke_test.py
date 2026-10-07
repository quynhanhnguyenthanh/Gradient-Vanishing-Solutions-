"""Bước 4 + 5: đo thời gian và kiểm tra pipeline chạy thông.

Chạy:  python smoke_test.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from config import Config
from train import run_experiment

N_RUNS_PLANNED = 130   # tổng số run của 3 pha


# ----------------------------------------------------------------------
# BƯỚC 0 - kiểm chứng: cấu hình mặc định phải tái hiện baseline tutorial
# ----------------------------------------------------------------------
def verify_baseline():
    print('=== Kiểm chứng baseline ===')
    cfg = Config(name='baseline', epochs=50, patience=50)  # tắt early stop
    res, _, init = run_experiment(cfg)

    loss_ok = abs(res['final_train_loss'] - 2.303) < 0.01
    acc_ok = abs(res['test_acc'] - 0.10) < 0.02
    print(f"  loss ≈ ln(10) = 2.303 ? {loss_ok}  ({res['final_train_loss']:.4f})")
    print(f"  test_acc ≈ 0.10 ?      {acc_ok}  ({res['test_acc']:.4f})")
    print(f"  decay_slope tại init: {init['decay_slope']:.3f} bậc/layer")
    print(f"  hệ số co mỗi layer  : {10 ** init['decay_slope']:.3f}")
    print(f"  effective_depth     : {init['effective_depth']}")

    if not (loss_ok and acc_ok):
        print('  !! LỆCH so với tutorial - kiểm tra chuẩn hóa dữ liệu và split')
    return res


# ----------------------------------------------------------------------
# BƯỚC 4 - đo thời gian, chốt quy mô
# ----------------------------------------------------------------------
def estimate_budget(runtime_s):
    total_h = runtime_s * N_RUNS_PLANNED / 3600
    print(f"\n=== Ngân sách ===")
    print(f"  1 run          : {runtime_s:.1f}s")
    print(f"  {N_RUNS_PLANNED} run: {total_h:.1f} giờ")
    if total_h > 4:
        print('  -> Quá lâu. Cắt Pha 3, hoặc giảm depth 25 trong Pha 2.')
    else:
        print('  -> Khả thi, giữ nguyên kế hoạch 3 pha.')


# ----------------------------------------------------------------------
# BƯỚC 5 - smoke test: 3 cấu hình, 1 seed, 5 epoch
# ----------------------------------------------------------------------
SMOKE = [
    Config(name='baseline', activation='sigmoid', init='normal_0.05'),
    Config(name='+relu',    activation='relu',    init='normal_0.05'),
    Config(name='+he',      activation='relu',    init='he'),
]


def smoke_test():
    print('\n=== Smoke test ===')
    rows, curves = [], {}
    for cfg in SMOKE:
        cfg.epochs, cfg.patience = 5, 5
        res, _, init = run_experiment(cfg)
        rows.append(res)
        curves[cfg.name] = init['grad_ratio']

    df = pd.DataFrame(rows)
    print(df[['name', 'test_acc', 'init_decay_slope',
              'init_grad_ratio_L1', 'runtime_s']].to_string(index=False))
    df.to_csv('smoke_summary.csv', index=False)

    # Hình chính: ‖∇W‖/‖W‖ theo layer, thang log
    plt.figure(figsize=(9, 5))
    for name, r in curves.items():
        ks = list(r.keys())
        plt.plot(ks, [r[k] for k in ks], 'o-', label=name)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.ylabel('‖∇W‖ / ‖W‖')
    plt.title('Gradient ratio theo layer (tại khởi tạo)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('smoke_gradient_ratio.png', dpi=120)
    plt.show()

    print('\nKiểm chứng: đường baseline phải dốc xuống mạnh nhất,')
    print('+relu và +he phải nằm cao hơn ở các layer đầu.')
    return df


if __name__ == '__main__':
    base = verify_baseline()
    estimate_budget(base['runtime_s'])
    smoke_test()
