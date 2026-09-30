"""Hệ chỉ số đo Gradient Vanishing.

Nguyên tắc:
  - Đại lượng chính là ‖∇W‖/‖W‖ vì nó tự chuẩn hóa theo quy mô layer.
  - decay_slope chỉ tính trên các layer cùng độ rộng, bỏ layer1 (784->w).
  - effective_depth báo cáo NHIỀU ngưỡng, không dựa vào một ngưỡng duy nhất.
  - Không kết luận từ một chỉ số: luôn báo cáo song song nhiều chỉ số.
"""

import numpy as np
import torch

THRESHOLDS = (0.001, 0.01, 0.05)   # 0.1% / 1% / 5%; 1% là primary


# ----------------------------------------------------------------------
# Gradient
# ----------------------------------------------------------------------
def grad_ratio_per_layer(model):
    """‖∇W‖/‖W‖ từng layer. Gọi sau backward(), trước step()."""
    out = {}
    for name, lin in model.weight_layers():
        if lin.weight.grad is not None:
            out[name] = (lin.weight.grad.norm()
                         / (lin.weight.norm() + 1e-12)).item()
    return out


def grad_abs_mean_per_layer(model):
    """Trung bình |∇W| - đại lượng tutorial dùng, giữ để đối chiếu Hình 13/14."""
    return {name: lin.weight.grad.abs().mean().item()
            for name, lin in model.weight_layers()
            if lin.weight.grad is not None}


def decay_slope(ratios, model):
    """Độ dốc hồi quy của log10(ratio) theo chỉ số layer.

    Chỉ dùng các layer w->w. Gần 0 = gradient được bảo toàn.
    Giá trị âm lớn = suy giảm nhanh theo cấp số nhân.
    """
    names = [n for n, _ in model.same_width_layers()] + ['output']
    vals = [ratios[n] for n in names if n in ratios]
    if len(vals) < 3:
        return float('nan')
    logv = np.log10(np.array(vals) + 1e-12)
    # trục x: 0 = layer nông nhất trong nhóm, tăng dần về output
    return float(np.polyfit(np.arange(len(logv)), logv, 1)[0])

def relative_gradient(ratios, model):
    """Tỉ số gradient giữa layer liền kề. Cho biết mỗi layer làm
    gradient co lại bao nhiêu lần."""
    names = [n for n, _ in model.weight_layers()]
    out = {}
    for a, b in zip(names[:-1], names[1:]):
        out[f'{a}->{b}'] = ratios[a] / (ratios[b] + 1e-12)
    return out

def effective_depth(ratios, thresholds=THRESHOLDS):
    """Số layer có ratio > threshold * ratio(output). Trả dict theo ngưỡng."""
    ref = ratios.get('output')
    if not ref:
        return {t: 0 for t in thresholds}
    return {t: sum(1 for v in ratios.values() if v > t * ref)
            for t in thresholds}


# ----------------------------------------------------------------------
# Mức cập nhật thực tế  
# ----------------------------------------------------------------------
def snapshot_weights(model):
    return {name: lin.weight.detach().clone()
            for name, lin in model.weight_layers()}


def update_ratio_per_layer(model, snapshot):
    """‖ΔW‖/‖W‖ sau một bước. Cho biết layer nào THỰC SỰ đang học.

    Gradient nhỏ mà update_ratio vẫn lớn -> optimizer đang bù mức cập nhật.
    """
    out = {}
    for name, lin in model.weight_layers():
        prev = snapshot[name]
        out[name] = ((lin.weight.detach() - prev).norm()
                     / (prev.norm() + 1e-12)).item()
    return out


# ----------------------------------------------------------------------
# Giải thích nguyên nhân: bão hòa và neuron chết
# ----------------------------------------------------------------------
class ActivationRecorder:
    """Ghi output của các hàm kích hoạt qua forward hook."""

    def __init__(self, model):
        self.store = {}
        self.handles = []
        for i, blk in enumerate(model.blocks):
            self.handles.append(
                blk.act.register_forward_hook(self._make_hook(f'layer{i + 1}'))
            )

    def _make_hook(self, name):
        def hook(_m, _inp, out):
            self.store[name] = out.detach()
        return hook

    def clear(self):
        self.store = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def saturation_rate(acts, threshold=0.01):
    """Tỉ lệ neuron sigmoid bão hòa: đạo hàm σ(1-σ) < threshold.

    Ngưỡng 0.01 ứng với σ ngoài khoảng ~[0.01, 0.99].
    """
    out = {}
    for name, a in acts.items():
        deriv = a * (1 - a)
        out[name] = (deriv < threshold).float().mean().item()
    return out


def dead_relu_rate(acts):
    """Tỉ lệ neuron ReLU có output = 0 trên toàn batch."""
    return {name: (a == 0).float().mean().item() for name, a in acts.items()}
