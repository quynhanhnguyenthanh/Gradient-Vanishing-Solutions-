# Gradient Vanishing: Đóng góp biên của từng giải pháp

Nghiên cứu thực nghiệm đo và so sánh hiệu quả của các giải pháp khắc phục Gradient Vanishing trên MLP sâu, dataset FashionMNIST.

---

## Câu hỏi nghiên cứu

> Khi được cô lập đúng cách, mỗi giải pháp khắc phục Gradient Vanishing đóng góp bao nhiêu, và thứ hạng giữa chúng có thay đổi khi độ sâu mạng tăng không?

Ba câu hỏi phụ:

1. Cô lập từng biến thì mỗi giải pháp khôi phục dòng gradient ở layer đầu đến mức nào?
2. Bỏ từng giải pháp khỏi cấu hình đầy đủ thì mất đi bao nhiêu? 
3. Thứ hạng có ổn định từ 3 đến 25 layer không?

---

## 5 giải pháp khảo sát

Mỗi giải pháp đại diện một cơ chế khác nhau:

| Cơ chế | Giải pháp | Can thiệp vào |
|---|---|---|
| Hàm kích hoạt | ReLU thay Sigmoid | Forward pass |
| Khởi tạo | He init | Điểm xuất phát |
| Chuẩn hóa | BatchNorm1d | Phân phối activation |
| Kiến trúc | Residual connection | Đường đi của gradient |
| Cập nhật trọng số | Adam thay SGD | Sau khi gradient đã tính |

---

## Hệ chỉ số

Đại lượng chính là **`‖∇W‖/‖W‖`** cho từng layer, không phải gradient norm thô - vì layer đầu nhận input 784 chiều nên có nhiều tham số hơn hẳn các hidden layer 128→128, so trực tiếp là so sai.

| Chỉ số | Cho biết |
|---|---|
| `grad_ratio` theo layer | Hình dạng suy giảm, vẽ trên thang log |
| `decay_slope` | Độ dốc hồi quy log10(ratio); chỉ tính trên layer cùng độ rộng |
| `effective_depth` | Số layer thực sự nhận gradient; báo cáo ở **3 ngưỡng** 0.1%/1%/5% |
| `update_ratio` = `‖ΔW‖/‖W‖` | Layer nào thực sự đang học — khác với gradient lớn hay nhỏ |
| `saturation_rate` | Tỉ lệ neuron sigmoid bão hòa - giải thích nguyên nhân |
| `dead_relu_rate` | Tỉ lệ neuron ReLU chết |

Đo tại **hai thời điểm**: lúc khởi tạo không phụ thuộc learning rate và sau mỗi epoch.

---

## Giao thức thí nghiệm

**Learning rate** - không dùng chung một giá trị (thang lr của SGD và Adam khác hẳn nhau), cũng không tune tùy ý. Mỗi cấu hình thử **cùng một lưới 4 giá trị** theo thang log, chọn giá trị tốt nhất trên validation, rồi chạy 3 seed tại giá trị đó.

**Seed** - 3 seed cho mọi cấu hình, **dùng chung một bộ seed** để so sánh trên cùng điều kiện khởi tạo. Chỉ kết luận một cấu hình tốt hơn khi chênh lệch lớn hơn rõ rệt so với std; nếu nằm trong khoảng std thì ghi "chưa phân biệt được".

**Ba pha**

| Pha | Nội dung | Ưu tiên |
|---|---|---|
| 1 | Ablation cô lập từng biến | Bắt buộc |
| 2 | Khảo sát theo độ sâu {3, 7, 15, 25} | Rất nên làm |
| 3 | Leave-one-out từ cấu hình đầy đủ | Nếu còn thời gian |

Pha 3 dùng leave-one-out thay vì cộng dồn vì kết quả cộng dồn phụ thuộc thứ tự thêm giải pháp.

---

## Cấu trúc

```
src/
  config.py    config dataclass, lưới learning rate, bộ seed
  model.py     build_mlp(cfg) — tham số hóa depth/activation/init/norm/residual
  metrics.py   hệ chỉ số đo vanishing
  train.py     nạp dữ liệu lên GPU, run_experiment(cfg)
  run_all.py   chạy toàn bộ 3 pha
notebooks/     phân tích và sinh hình
results/
  summary.csv   1 dòng mỗi run
  figures/
docs/
  outline.md
```

## Chạy

```bash
pip install -r requirements.txt
python src/smoke_test.py      # kiểm chứng baseline + smoke test
python src/run_all.py         # chạy toàn bộ thí nghiệm
```

Trên Colab:

```python
!git clone https://github.com/<user>/gradient-vanishing-topic-team.git
%cd gradient-vanishing-topic-team
import sys; sys.path.append('src')
```

---

## Tiến độ

- [x] Tái hiện baseline tutorial - train loss 2.3037 (≈ ln 10 = 2.3026), test acc 0.10
- [x] Tái hiện gradient flow và heatmap layer × epoch
- [x] Tối ưu pipeline (1003s → mục tiêu <60s mỗi run)
- [x] `metrics.py` đầy đủ
- [x] Pha 1 - ablation
- [ ] Pha 2 - depth scaling
- [ ] Pha 3 - leave-one-out

---

## Paper tham khảo

1. Glorot & Bengio (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
2. Pascanu et al. (2013). On the difficulty of training recurrent neural networks. *ICML*.
3. He et al. (2015). Delving Deep into Rectifiers. *ICCV*.
4. Ioffe & Szegedy (2015). Batch Normalization. *ICML*.
5. He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
6. Guo et al. (2024). Take A Shortcut Back. *NeurIPS*. arXiv:2401.04486.

---

## License

MIT
