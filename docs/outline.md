# Outline - Thiết kế nghiên cứu

**Topic Team AIO2026** · Subtopic: Các giải pháp khắc phục Gradient Vanishing

Tài liệu này mô tả thiết kế thí nghiệm và lý do đằng sau từng lựa chọn. 
---

## 1. Câu hỏi nghiên cứu

> Khi được cô lập đúng cách, mỗi giải pháp khắc phục Gradient Vanishing đóng góp bao nhiêu, và thứ hạng giữa chúng có thay đổi khi độ sâu mạng tăng không?

Ba câu hỏi phụ:

1. Cô lập từng biến thì mỗi giải pháp khôi phục dòng gradient đến mức nào?
2. Bỏ từng giải pháp khỏi cấu hình đầy đủ thì mất đi bao nhiêu?
3. Thứ hạng có ổn định từ 3 đến 25 layer không?

---

## 2. Vị trí so với tài liệu có sẵn

Tutorial *Vanishing Gradient* (AI VIET NAM, AIO2025) dựng baseline MLP 7 layer, 128 node, Sigmoid, init `N(0, 0.05)`, SGD trên FashionMNIST, và cài 9 giải pháp. Bốn khoảng trống:

| # | Hạn chế | Hệ quả |
|---|---|---|
| 1 | Chỉ chạy một seed cố định | Không phân biệt chênh lệch thật với nhiễu |
| 2 | Báo cáo kết quả gộp của 8 cải tiến | Không biết giải pháp nào đóng góp bao nhiêu |
| 3 | Một số cấu hình đổi nhiều biến cùng lúc | Không cô lập được ảnh hưởng từng biến |
| 4 | Độ sâu cố định ở 7 layer | Không biết thứ hạng có đổi theo độ sâu không |

Nghiên cứu này giữ nguyên dataset, kiến trúc nền và baseline của tutorial để kết quả so sánh trực tiếp được, đồng thời bổ sung: cô lập biến đúng cách, đa seed với giao thức rõ ràng, khảo sát theo độ sâu, hệ chỉ số định lượng, và phân tích leave-one-out.

---

## 3. Phạm vi: 5 giải pháp cốt lõi

Thu hẹp từ 9 xuống 5, mỗi giải pháp đại diện **một cơ chế khác nhau**. Lý do không phải chi phí tính toán mà là thời gian phân tích.

| Cơ chế | Giải pháp | Can thiệp vào |
|---|---|---|
| Hàm kích hoạt | ReLU thay Sigmoid | Forward pass |
| Khởi tạo trọng số | He init | Điểm xuất phát |
| Chuẩn hóa | BatchNorm1d | Phân phối activation |
| Kiến trúc | Residual connection mỗi 2 layer | Đường đi của gradient |
| Cập nhật trọng số | Adam thay SGD | Sau khi gradient đã tính xong |

**Nguyên tắc loại bỏ:** các giải pháp trùng cơ chế (biến thể của ReLU) chỉ giữ một đại diện; các giải pháp không nhắm vào vanishing (gradient clipping xử lý exploding) bị loại.

**Cấu hình bổ sung `+relu_he`.** He init được suy ra với giả định ReLU triệt tiêu một nửa phân phối. Nên `+relu` (ReLU + init nhỏ) và `+he` (sigmoid + He init) đều là cấu hình không có cơ sở lý thuyết. Giữ cả ba để kiểm chứng xem hai biến này có độc lập không — thiết kế OFAT thuần túy giả định chúng độc lập, và đây là cách kiểm tra giả định đó.

**Không đưa vào bảng chính:** Layer-wise Training và Progressive Training thay đổi cả quy trình huấn luyện nhiều giai đoạn nên không có "một biến" để cô lập.

---

## 4. Hệ chỉ số

### 4.1 Đại lượng chính: `‖∇W‖/‖W‖`

Không dùng gradient norm thô vì layer1 nhận input 784 chiều, có nhiều tham số hơn hẳn các hidden layer 128×128, nên gradient norm của nó tự nhiên lớn hơn. Đại lượng tự chuẩn hóa theo quy mô layer so sánh được giữa mọi layer và mọi cấu hình.

### 4.2 Đo cả activation gradient

Weight gradient `∂L/∂W_l = δ_l · a_{l-1}ᵀ` bị lẫn độ lớn activation đầu vào, nên gradient nhỏ chưa chắc do vanishing. Activation gradient `δ_l = ∂L/∂h_l` là tín hiệu thực sự được truyền ngược, phản ánh cơ chế trực tiếp hơn.

Đo cả hai: activation gradient cho **cơ chế**, weight gradient cho **hệ quả** lên việc học. Nếu hai đại lượng cho hình dạng khác nhau, đó là một phát hiện phải báo cáo.

Với BatchNorm có cảnh báo riêng: output không đổi khi nhân `W` với hằng số, nên `‖∇W‖` tỉ lệ nghịch với `‖W‖`. Phải kiểm tra lại bảng so sánh cho cấu hình đó.

### 4.3 Chỉ số tóm tắt

| Chỉ số | Định nghĩa | Cho biết |
|---|---|---|
| `decay_slope` | độ dốc hồi quy `log10(ratio)` theo chỉ số layer, chỉ trên layer cùng độ rộng | Mức suy giảm mỗi layer |
| `R²` | độ khớp của đường hồi quy trên | Suy giảm có thật theo cấp số nhân không. R² thấp → slope đơn lẻ mất ý nghĩa |
| `effective_depth` | số layer có ratio > ngưỡng × ratio(output), báo cáo ở **3 ngưỡng** 0.1%/1%/5% | Bao nhiêu layer thực sự nhận gradient |
| `update_ratio` | `‖ΔW‖/‖W‖` sau một bước | Layer nào thực sự được cập nhật |
| `drift` | `‖W_t − W_0‖/‖W_0‖` | Layer nào đã dịch chuyển khỏi điểm khởi tạo |
| `act_norm` | RMS activation từng layer | Tín hiệu forward có tắt dần không |
| `sparsity` / `dead_relu` | tỉ lệ output = 0 mỗi mẫu / tỉ lệ neuron = 0 mọi mẫu | Hai hiện tượng khác nhau, phải tách |

**Vì sao ba ngưỡng cho `effective_depth`:** một ngưỡng duy nhất là tùy tiện. Nếu thứ hạng giữa các cấu hình không đổi qua cả ba ngưỡng thì kết luận vững; nếu đổi, đó là phát hiện phải nêu.

**Vì sao cần `drift`:** khi `decay_slope` không đổi theo thời gian, có hai cách giải thích — vanishing là trạng thái ổn định, hoặc model không học gì nên trọng số không nhúc nhích. `drift` phân biệt được hai trường hợp này.

### 4.4 Thời điểm đo

Đo tại **lúc khởi tạo** (trước bước cập nhật đầu tiên) và **trong quá trình huấn luyện** (mỗi `measure_every` epoch).

Gradient đo tại khởi tạo không phụ thuộc learning rate vì chưa có bước cập nhật nào. Nên chỉ số chính về vanishing không bị ảnh hưởng bởi cách chọn lr; lr chỉ ảnh hưởng đến loss, accuracy và tốc độ hội tụ.

---

## 5. Giao thức

### 5.1 Thiết lập chung

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Dataset | FashionMNIST, chia 90/10 train/val | Theo tutorial |
| Kiến trúc | MLP, hidden width 128 | Trừ thí nghiệm đổi width |
| Baseline | 7 hidden layer, Sigmoid, init `N(0, 0.05)`, SGD | Theo tutorial |
| Batch size | 256 | Tutorial không nêu giá trị |
| Epoch | 50, early stopping patience 15 | Tutorial dùng 100 cố định |
| Nạp dữ liệu | Toàn bộ dataset trên GPU, không DataLoader | Giảm runtime từ 1003s xuống ~27s |

Patience đặt rộng vì theo Glorot & Bengio, unit bão hòa có thể tự thoát nhưng rất chậm, tạo plateau dài trước khi loss giảm tiếp. Early stopping quá gắt sẽ cắt đúng lúc mô hình sắp thoát.

### 5.2 Learning rate: cùng ngân sách tune

Không cố định một giá trị chung vì thang lr của SGD và Adam khác nhau hoàn toàn. Không tune tùy ý vì cấu hình được tune kỹ hơn sẽ có lợi thế không công bằng.

**Giao thức:** mỗi cấu hình thử cùng một lưới 4 giá trị theo thang log, chọn giá trị tốt nhất trên validation (1 seed, 20 epoch), rồi chạy 3 seed tại giá trị đó.

```
SGD : [1e-3, 1e-2, 1e-1, 5e-1]
Adam: [1e-5, 1e-4, 1e-3, 1e-2]
```

Mọi lr làm tràn số (`loss` hoặc `acc` không hữu hạn) bị loại khỏi lựa chọn. Khi hai lr cho kết quả chênh nhau dưới `1e-4` thì giữ lr nhỏ hơn.

### 5.3 Seed

- 3 seed cho mọi cấu hình: `[42, 43, 44]`
- **Dùng chung một bộ seed** để các cấu hình được so sánh trên cùng điều kiện khởi tạo và cùng thứ tự dữ liệu
- Báo cáo `mean ± std`
- **Chỉ kết luận A tốt hơn B khi chênh lệch vượt 2σ**; nằm trong σ thì ghi "chưa phân biệt được"

---

## 6. Sáu pha thí nghiệm

| Pha | Nội dung | Cấu hình | Mục đích |
|---|---|---|---|
| 1 | Ablation cô lập từng biến, depth 7 | 7 | Đóng góp của từng giải pháp |
| 2 | Khảo sát độ sâu {3, 7, 15, 25} | 28 | Thứ hạng có đổi theo độ sâu không |
| 2b | Lấp vùng chuyển tiếp {4, 5, 6, 8, 9, 10, 11, 13} | 20 | Xác định ngưỡng chính xác |
| 2c | Kiểm chứng bằng width {32, 64, 256} × depth {3, 5, 7} | 9 | Ngưỡng có phụ thuộc kiến trúc không |
| 3 | Leave-one-out từ cấu hình đầy đủ | 6 | Đóng góp biên khi đã có giải pháp khác |
| 3b | Đo mỗi epoch ở ba chế độ | 3 | Slope có biến đổi theo thời gian không |

**Tổng: 72 cấu hình × 3 seed = 216 run.**

**Vì sao leave-one-out thay vì cộng dồn:** kết quả cộng dồn phụ thuộc thứ tự thêm — giải pháp thêm vào sau luôn có vẻ đóng góp ít hơn vì phần cải thiện đã bị các giải pháp trước "lấy mất". Leave-one-out không có vấn đề này.

**Vì sao cần pha 2c:** ngưỡng tìm được ở pha 2 và 2b chỉ suy ra từ việc đổi `depth`. Nếu nó là quy luật thật thì đổi `width` — một biến hoàn toàn độc lập — cũng phải tuân theo. Đây là phép kiểm chứng chéo.

---

## 7. Cấu trúc code

```
src/
  config.py     Config dataclass, lưới learning rate, bộ seed
  model.py      build_mlp(cfg) — tham số hóa depth/width/activation/init/norm/residual
  metrics.py    hệ chỉ số đo vanishing
  train.py      nạp dữ liệu lên GPU, run_experiment(cfg)
  run_all.py    chạy các pha, resume được khi mất kết nối
  analysis.py   đọc results/ và sinh bảng
  make_figures.py  sinh 6 hình cuối
notebooks/
  01_baseline.ipynb   tái hiện baseline tutorial, xác nhận
  02_analysis.ipynb   bảng và hình cho báo cáo
results/
  summary.csv   1 dòng mỗi run
  lr_sweep.csv  kết quả tune learning rate
  raw/          1 JSON mỗi run: gradient theo layer, theo epoch
  figures/
docs/
  outline.md    tài liệu này
```

Mọi kết quả ghi tăng dần vào `summary.csv` nên mất kết nối không mất dữ liệu; chạy lại sẽ bỏ qua các run đã xong.

---

## 8. Rủi ro và phương án

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Thời gian chạy vượt dự kiến | Pha 2 chưa xong giữa tuần 3 | Giảm `measure_every`, cắt pha 3 |
| Chênh lệch nằm trong sai số | σ chồng lấn | Tăng seed cho cặp then chốt; nếu vẫn chồng thì báo cáo "chưa phân biệt được" |
| lr được chọn không bền | loss thành NaN giữa chừng | Loại lr diverge khỏi lựa chọn; nêu trong hạn chế |
| Residual làm tỉ số gradient khó diễn giải | Đường log không còn tuyến tính | Phân tích riêng, nêu rõ gradient có đường đi tắt qua phép cộng |

---

## 9. Ghi nguồn

Code baseline (kiến trúc MLP, vòng lặp huấn luyện, cấu hình dữ liệu) kế thừa từ *Tutorial: Vanishing Gradient*, AI VIET NAM, AI Course 2025 (Nguyễn Thọ Anh Khoa, Dương Đình Thắng, Nguyễn Phúc Thịnh, Nguyễn Anh Khôi).

Phần tự phát triển: hệ chỉ số đo, tham số hóa model, giao thức learning rate và seed, thiết kế sáu pha thí nghiệm, và toàn bộ phân tích.
