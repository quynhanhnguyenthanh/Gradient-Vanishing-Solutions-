"""MLP tham số hóa. Gọi với Config mặc định phải ra ĐÚNG baseline tutorial."""

import torch.nn as nn


def make_activation(name):
    return {
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
    }[name]()


def apply_init(module, scheme, activation):
    if not isinstance(module, nn.Linear):
        return
    if scheme == 'normal_0.05':
        nn.init.normal_(module.weight, 0.0, 0.05)
    elif scheme == 'normal_1.0':
        nn.init.normal_(module.weight, 0.0, 1.0)
    elif scheme == 'xavier':
        nn.init.xavier_uniform_(module.weight)   # Glorot dùng uniform
    elif scheme == 'he':
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    else:
        raise ValueError(f'init không hợp lệ: {scheme}')
    nn.init.constant_(module.bias, 0.0)


class Block(nn.Module):
    """Linear -> (Norm) -> Activation. Đơn vị nhỏ nhất của mạng."""

    def __init__(self, in_dim, out_dim, activation, norm):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim) if norm == 'batchnorm' else None
        self.act = make_activation(activation)

    def forward(self, x):
        x = self.lin(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class MLP(nn.Module):
    def __init__(self, cfg, input_dims=784, output_dims=10):
        super().__init__()
        self.cfg = cfg
        w = cfg.width

        # block[0]: 784 -> w   (kích thước KHÁC các block sau, nhớ khi tính metric)
        # block[1..depth-1]: w -> w
        self.blocks = nn.ModuleList(
            [Block(input_dims, w, cfg.activation, cfg.norm)]
            + [Block(w, w, cfg.activation, cfg.norm) for _ in range(cfg.depth - 1)]
        )
        self.output = nn.Linear(w, output_dims)

        for m in self.modules():
            apply_init(m, cfg.init, cfg.activation)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.blocks[0](x)                     # block đầu không tham gia residual

        i = 1
        while i < len(self.blocks):
            if self.cfg.residual and i + 1 < len(self.blocks):
                identity = x
                x = self.blocks[i](x)
                x = self.blocks[i + 1](x)
                x = x + identity                  # cộng SAU activation, theo tutorial
                i += 2
            else:
                x = self.blocks[i](x)
                i += 1

        return self.output(x)

    def weight_layers(self):
        """Trả [(tên, module Linear)] theo thứ tự input -> output."""
        out = [(f'layer{i + 1}', b.lin) for i, b in enumerate(self.blocks)]
        out.append(('output', self.output))
        return out

    def same_width_layers(self):
        """Chỉ các layer w->w. Dùng cho decay_slope để loại ảnh hưởng layer1."""
        return [(f'layer{i + 1}', b.lin)
                for i, b in enumerate(self.blocks) if i >= 1]


def build_mlp(cfg):
    return MLP(cfg)

