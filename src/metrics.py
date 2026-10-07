"""Hệ chỉ số đo Gradient Vanishing.

Nguyên tắc (theo góp ý của TA):
  - Đo CẢ HAI loại gradient:
      activation gradient δ_l = ∂L/∂h_l  -> phản ánh CƠ CHẾ vanishing
      weight gradient ∂L/∂W_l = δ_l·a_{l-1}ᵀ -> phản ánh HỆ QUẢ lên việc học
    Weight gradient nhỏ chưa chắc do vanishing: nó bị lẫn độ lớn của
    activation đầu vào. Với BatchNorm, ‖∇W‖ còn tỉ lệ nghịch với ‖W‖.
  - Chuẩn hóa theo số phần tử (RMS thay vì norm) để layer rộng hơn
    không trông như có gradient lớn hơn.
  - decay_slope chỉ tính trên các layer cùng độ rộng, bỏ layer1 (784->w).
  - effective_depth báo cáo NHIỀU ngưỡng, không dựa một ngưỡng duy nhất.
  - Phân biệt sparsity (output=0 mỗi mẫu) với dead-ReLU (neuron=0 mọi mẫu).
"""

import numpy as np
import torch

THRESHOLDS = (0.001, 0.01, 0.05)   # 0.1% / 1% / 5%; 1% là primary


# ======================================================================
# 1. WEIGHT GRADIENT  -  hệ quả lên việc học
# ======================================================================
def grad_ratio_per_layer(model):
    """‖∇W‖/‖W‖ từng layer. Gọi sau backward(), trước step().

    Tự chuẩn hóa theo quy mô layer nên so sánh được giữa các layer
    kích thước khác nhau. Nhưng vẫn là weight gradient: xem cảnh báo
    về BatchNorm ở docstring đầu file.
    """
    out = {}
    for name, lin in model.weight_layers():
        if lin.weight.grad is not None:
            out[name] = (lin.weight.grad.norm()
                         / (lin.weight.norm() + 1e-12)).item()
    return out


def grad_rms_per_layer(model):
    """RMS của ∇W từng layer, chuẩn hóa theo SỐ PHẦN TỬ."""
    return {name: lin.weight.grad.pow(2).mean().sqrt().item()
            for name, lin in model.weight_layers()
            if lin.weight.grad is not None}


def grad_abs_mean_per_layer(model):
    """Trung bình |∇W| - đại lượng tutorial dùng, giữ để đối chiếu."""
    return {name: lin.weight.grad.abs().mean().item()
            for name, lin in model.weight_layers()
            if lin.weight.grad is not None}


def effective_update_ratio(model, lr):
    """lr · ‖∇W‖/‖W‖ - chỉ số thực tế nhất cho biết layer có THẬT SỰ học,
    không phụ thuộc thang đo của W.

    CHÚ Ý: với Adam công thức này KHÔNG chính xác, vì Adam tự rescale
    bước cập nhật. Với Adam phải dùng update_ratio_per_layer đo trực tiếp.
    """
    return {name: lr * (lin.weight.grad.norm()
                        / (lin.weight.norm() + 1e-12)).item()
            for name, lin in model.weight_layers()
            if lin.weight.grad is not None}


# ======================================================================
# 2. ACTIVATION GRADIENT  -  cơ chế vanishing
# ======================================================================
def act_grad_rms(grads):
    """RMS của δ_l = ∂L/∂h_l từng layer.

    Đây là tín hiệu thực sự được truyền ngược: δ_l = J_{l+1}ᵀ δ_{l+1}.
    Vanishing nghĩa là tích các Jacobian co dần về 0, và đại lượng này
    cho thấy điều đó trực tiếp, không bị lẫn độ lớn activation đầu vào.
    """
    return {name: g.pow(2).mean().sqrt().item() for name, g in grads.items()}


def act_grad_relative(grads):
    """Tỉ số δ giữa hai layer liền kề, theo chiều backward.
    Giá trị ~8 nghĩa là tín hiệu co 8 lần khi qua layer đó."""
    rms = act_grad_rms(grads)
    names = sorted(rms.keys(), key=lambda s: int(s.replace('layer', '')))
    out = {}
    for a, b in zip(names[:-1], names[1:]):
        out[f'{b}->{a}'] = rms[b] / (rms[a] + 1e-12)
    return out


# ======================================================================
# 3. SUY GIẢM THEO ĐỘ SÂU
# ======================================================================
def decay_slope(ratios, model):
    """Độ dốc hồi quy của log10(ratio) theo chỉ số layer.

    Chỉ dùng các layer w->w để loại ảnh hưởng của layer1 (784->w).
    Gần 0 = gradient được bảo toàn. Dương lớn = suy giảm nhanh về input.
    """
    names = [n for n, _ in model.same_width_layers()] + ['output']
    vals = [ratios[n] for n in names if n in ratios]
    if len(vals) < 3:
        return float('nan')
    logv = np.log10(np.array(vals) + 1e-12)
    return float(np.polyfit(np.arange(len(logv)), logv, 1)[0])


def decay_slope_with_r2(ratios, model):
    """Trả (slope, r2). R2 thấp nghĩa là suy giảm KHÔNG theo cấp số nhân,
    lúc đó slope đơn lẻ mất ý nghĩa và phải xem toàn bộ profile."""
    names = [n for n, _ in model.same_width_layers()] + ['output']
    vals = [ratios[n] for n in names if n in ratios]
    if len(vals) < 3:
        return float('nan'), float('nan')
    y = np.log10(np.array(vals) + 1e-12)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return float(slope), float(r2)


def relative_gradient(ratios, model):
    """Tỉ số ‖∇W‖/‖W‖ giữa hai layer liền kề, theo chiều backward."""
    names = [n for n, _ in model.weight_layers()]
    out = {}
    for a, b in zip(names[:-1], names[1:]):
        out[f'{b}->{a}'] = ratios[b] / (ratios[a] + 1e-12)
    return out


def effective_depth(ratios, thresholds=THRESHOLDS):
    """Số layer có ratio > threshold * ratio(output). Trả dict theo ngưỡng."""
    ref = ratios.get('output')
    if not ref:
        return {t: 0 for t in thresholds}
    return {t: sum(1 for v in ratios.values() if v > t * ref)
            for t in thresholds}


# ======================================================================
# 4. MỨC CẬP NHẬT THỰC TẾ
# ======================================================================
def snapshot_weights(model):
    return {name: lin.weight.detach().clone()
            for name, lin in model.weight_layers()}


def update_ratio_per_layer(model, snapshot):
    """‖ΔW‖/‖W‖ sau MỘT bước. Layer nào thực sự được cập nhật.

    Gradient nhỏ mà update_ratio vẫn lớn -> optimizer đang bù mức cập nhật.
    Đây là đại lượng đúng để dùng với Adam.
    """
    out = {}
    for name, lin in model.weight_layers():
        prev = snapshot[name]
        out[name] = ((lin.weight.detach() - prev).norm()
                     / (prev.norm() + 1e-12)).item()
    return out


def weight_drift(model, w0):
    """‖W_t − W_0‖/‖W_0‖ từng layer: tích lũy từ điểm khởi tạo.

    Phân biệt hai cách giải thích khi slope không đổi:
      drift đáng kể mọi layer -> vanishing là trạng thái ổn định
      drift ~ 0 ở layer đầu   -> model gần như không học gì
    """
    out = {}
    for name, lin in model.weight_layers():
        out[name] = ((lin.weight.detach() - w0[name]).norm()
                     / (w0[name].norm() + 1e-12)).item()
    return out


# ======================================================================
# 5. NGUYÊN NHÂN: bão hòa, tính thưa, neuron chết
# ======================================================================
class ActivationRecorder:
    """Ghi output và (tùy chọn) gradient của các hàm kích hoạt.

    capture_grad=True gắn backward hook lên tensor output để lấy
    δ_l = ∂L/∂h_l, đại lượng phản ánh cơ chế vanishing trực tiếp.
    """

    def __init__(self, model, capture_grad=False):
        self.store = {}
        self.grads = {}
        self.handles = []
        for i, blk in enumerate(model.blocks):
            self.handles.append(
                blk.act.register_forward_hook(
                    self._make_hook(f'layer{i + 1}', capture_grad))
            )

    def _make_hook(self, name, capture_grad):
        def hook(_m, _inp, out):
            self.store[name] = out.detach()
            if capture_grad and out.requires_grad:
                out.register_hook(
                    lambda g, n=name: self.grads.__setitem__(n, g.detach()))
        return hook

    def clear(self):
        self.store, self.grads = {}, {}

    def remove(self):
        for h in self.handles:
            h.remove()


def saturation_rate(acts, threshold=0.01):
    """Tỉ lệ neuron sigmoid bão hòa: đạo hàm σ(1-σ) < threshold.
    Ngưỡng 0.01 ứng với σ ngoài khoảng ~[0.01, 0.99].

    CHÚ Ý: công thức này chỉ đúng cho sigmoid. Với tanh đạo hàm là 1-a².
    """
    out = {}
    for name, a in acts.items():
        deriv = a * (1 - a)
        out[name] = (deriv < threshold).float().mean().item()
    return out


def sparsity_rate(acts):
    """Tỉ lệ output bằng 0, tính trên MỖI MẪU. Đây là tính thưa, có ích."""
    return {name: (a == 0).float().mean().item() for name, a in acts.items()}


def dead_relu_rate(acts):
    """Tỉ lệ NEURON không kích hoạt với BẤT KỲ mẫu nào. Neuron chết thật.

    Khác sparsity: sparsity cao vẫn có thể tốt (biểu diễn thưa),
    còn dead-ReLU cao là mất hẳn capacity.
    """
    out = {}
    for name, a in acts.items():
        never_active = (a.abs().sum(dim=0) == 0)   # cộng theo chiều batch
        out[name] = never_active.float().mean().item()
    return out


def act_norm_per_layer(acts):
    """RMS của activation từng layer. Kiểm tra xem tín hiệu có phình lên
    theo độ sâu không - đáng chú ý với residual + sigmoid, vì F(x) luôn
    dương nên x cộng dồn đơn điệu."""
    return {name: a.pow(2).mean().sqrt().item() for name, a in acts.items()}
