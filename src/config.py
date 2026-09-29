from dataclasses import dataclass, asdict


@dataclass
class Config:
    name: str = 'baseline'

    # kiến trúc
    depth: int = 7                    # số hidden layer
    width: int = 128
    activation: str = 'sigmoid'       # sigmoid | tanh | relu | leaky_relu
    init: str = 'normal_0.05'         # normal_0.05 | normal_1.0 | xavier | he
    norm: str = 'none'                # none | batchnorm
    residual: bool = False

    # huấn luyện
    optimizer: str = 'sgd'            # sgd | adam
    lr: float = 1e-2
    batch_size: int = 256
    epochs: int = 50
    patience: int = 15                # early stopping; đặt rộng vì plateau
    min_delta: float = 1e-4

    seed: int = 42

    def to_dict(self):
        return asdict(self)


# Bộ seed dùng chung cho MỌI cấu hình (yêu cầu của TA: giảm nhiễu khi so sánh)
SEEDS = [42, 43, 44]

# Lưới learning rate, cùng ngân sách tune cho mọi cấu hình
LR_GRID = {
    'sgd':  [1e-3, 1e-2, 1e-1, 5e-1],
    'adam': [1e-5, 1e-4, 1e-3, 1e-2],
}

