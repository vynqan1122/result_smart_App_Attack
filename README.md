# Result Smart App Attack — Hướng dẫn đọc kết quả thực nghiệm

Repo này lưu mã nguồn và kết quả thực nghiệm cho bài toán đánh giá độ dễ bị tấn công đối kháng của mô hình nhận diện ảnh trong kịch bản SmartAppAttack. Mục tiêu chính của README này là giúp người đọc mới mở repo có thể hiểu nhanh:

- Thí nghiệm đang làm gì.
- Kết quả nằm ở đâu.
- Các cột trong file kết quả có ý nghĩa gì.
- Cách đọc chỉ số `attack success rate`.
- File code nào dùng cho bước nào trong pipeline.

---

## 1. Ý tưởng thực nghiệm

Thực nghiệm xây dựng bài toán phân loại nhị phân giữa:

- `stop`: lớp mục tiêu, được gán nhãn số `1` trong dữ liệu dạng thư mục của Keras.
- `non_stop`: lớp không phải mục tiêu, được gán nhãn số `0`.

Sau khi huấn luyện mô hình, repo dùng các thuật toán tấn công đối kháng để tạo ảnh nhiễu/adversarial image. Sau đó ảnh đối kháng được đưa lại vào mô hình nạn nhân dạng TFLite để kiểm tra mô hình còn dự đoán đúng lớp mục tiêu hay đã bị đánh lừa.

Nói ngắn gọn:

```text
Dữ liệu ảnh gốc
    ↓
Tạo bộ dữ liệu nhị phân stop / non_stop
    ↓
Huấn luyện mô hình với các backbone khác nhau
    ↓
Chuyển mô hình sang TFLite
    ↓
Sinh ảnh đối kháng bằng FGSM / CAN / CW / PGD / MI-FGSM
    ↓
Đánh giá Attack Success Rate trên mô hình TFLite
    ↓
Tổng hợp kết quả ra CSV và biểu đồ
```

---

## 2. Đọc kết quả ở đâu?

Các file quan trọng nhất để xem kết quả là:

| Đường dẫn | Vai trò |
|---|---|
| `appendix_outputs/attack_summary.csv` | File tổng hợp kết quả chính, có nhiều thông tin nhất. Nên đọc file này đầu tiên. |
| `appendix_outputs/attack_success_summary.png` | Biểu đồ cột trực quan hóa `mean_success_rate` của các thí nghiệm. |
| `attack_summary.csv` | Bản tóm tắt ngắn hơn ở thư mục gốc, có vẻ là kết quả chạy nhanh hoặc kết quả trung gian. |
| `figures/figures/` | Chứa hình loss/accuracy theo từng dataset, backbone, chế độ huấn luyện và baseline. |
| `results_json/` | Là nơi script dự kiến lưu kết quả chi tiết từng lần chạy. Nếu không thấy thư mục này trong repo, có thể do chưa được commit đầy đủ. |
| `adv_examples/` | Là nơi script dự kiến lưu ảnh đối kháng sau khi chạy attack. |

Khuyến nghị: khi viết báo cáo hoặc phụ lục, nên ưu tiên dùng `appendix_outputs/attack_summary.csv` và `appendix_outputs/attack_success_summary.png`, vì đây là phần tổng hợp dành cho đọc kết quả.

---

## 3. Ý nghĩa các cột trong `attack_summary.csv`

Trong file `appendix_outputs/attack_summary.csv`, mỗi dòng tương ứng với một cấu hình thực nghiệm.

| Cột | Ý nghĩa |
|---|---|
| `dataset` | Bộ dữ liệu được dùng, ví dụ `GTSRB`, `CIFAR10`, `FLOWERS`. |
| `backbone` | Kiến trúc mạng dùng làm backbone, ví dụ `MobileNetV2`, `InceptionV3`, `ResNet50V2`. |
| `training_mode` | Cách huấn luyện mô hình: `feature_extraction` hoặc `fine_tuning`. |
| `baseline` | Phương pháp/baseline được đánh giá: `PMA`, `BAMA`, `E-BAMA`. |
| `attack_algo` | Thuật toán tấn công: `FGSM`, `CAN`, `CW`, `PGD`, `MIFGSM`. |
| `model_name` | Tên mô hình cụ thể được đem đi đánh giá. |
| `epsilon` | Mức nhiễu/tấn công. Trong code, ảnh được xử lý theo thang pixel `0–255`, nên `epsilon=8` nghĩa là ngân sách nhiễu khoảng 8 đơn vị pixel. |
| `num_images` | Số ảnh được dùng để đánh giá trong cấu hình đó. |
| `mean_success_rate` | Tỷ lệ tấn công thành công trung bình. Đây là cột quan trọng nhất. |
| `min_success_rate` | Tỷ lệ tấn công thành công thấp nhất qua các vòng chạy. |
| `max_success_rate` | Tỷ lệ tấn công thành công cao nhất qua các vòng chạy. |
| `path` | Đường dẫn tới file JSON chi tiết tương ứng với dòng kết quả. |

---

## 4. Cách hiểu Attack Success Rate

`Attack Success Rate` hay `ASR` là tỷ lệ ảnh ban đầu thuộc lớp mục tiêu nhưng sau khi bị tấn công thì mô hình không còn dự đoán là lớp mục tiêu nữa.

Ví dụ:

- `mean_success_rate = 1.0`: tấn công thành công 100%, mô hình bị đánh lừa trên toàn bộ ảnh được xét.
- `mean_success_rate = 0.84`: tấn công thành công 84%.
- `mean_success_rate = 0.0`: tấn công không làm đổi được dự đoán của mô hình trong cấu hình đó.

Khi đọc kết quả:

- ASR càng cao → attack càng hiệu quả, mô hình càng dễ bị đánh lừa.
- ASR càng thấp → attack kém hiệu quả hơn trong cấu hình hiện tại, hoặc mô hình có vẻ ổn định hơn trước kiểu attack đó.

Lưu ý: không nên kết luận mô hình “an toàn tuyệt đối” chỉ vì một attack có ASR thấp. Một attack khác, epsilon khác, số bước khác hoặc tập ảnh lớn hơn có thể cho kết quả khác.

---

## 5. Kết quả nổi bật đang có trong repo

Phần kết quả rõ nhất trong repo hiện tại là nhóm thực nghiệm trên:

```text
Dataset: GTSRB
Backbone: MobileNetV2
Training mode: feature_extraction
Số ảnh đánh giá: 50
```

Bảng dưới đây tóm tắt các kết quả chính từ `appendix_outputs/attack_summary.csv`:

| Baseline | FGSM | CAN | CW | PGD | MI-FGSM | Nhận xét nhanh |
|---|---:|---:|---:|---:|---:|---|
| PMA | 10.00% | 1.32% | 10.00% | Chưa thấy trong summary | Chưa thấy trong summary | PMA có ASR thấp trong nhóm kết quả này. |
| BAMA | 84.00% | 1.20% | 90.00% | 100.00% | 100.00% | BAMA bị tấn công rất mạnh bởi FGSM, CW, PGD và MI-FGSM. |
| E-BAMA | 96.00% | 0.00% | 92.00% | 100.00% | 100.00% | E-BAMA cũng rất dễ bị đánh lừa bởi FGSM, CW, PGD và MI-FGSM trong cấu hình này. |

Nhận xét chính:

1. `PGD` và `MI-FGSM` là hai attack mạnh nhất trong phần kết quả feature extraction, với ASR đạt 100% trên BAMA và E-BAMA.
2. `FGSM` dù là attack một bước nhưng vẫn đạt ASR rất cao trên BAMA và E-BAMA.
3. `CW` cũng cho ASR cao trên BAMA và E-BAMA.
4. `CAN` gần như không hiệu quả trong các cấu hình đã tổng hợp, ASR xấp xỉ 0%.
5. PMA có ASR thấp hơn rõ rệt so với BAMA và E-BAMA trong nhóm kết quả GTSRB + MobileNetV2 + feature extraction.

---

## 6. Nhận xét về nhóm fine-tuning

Trong phần `fine_tuning`, repo có nhiều kết quả theo số lớp được fine-tune, ví dụ các model có hậu tố:

```text
_10_sim
_20_sim
_30_sim
_40_sim
_50_sim
_60_sim
_stop_sim
```

Cách hiểu các hậu tố này:

- `_10_sim`, `_20_sim`, ..., `_60_sim`: các biến thể fine-tuning với số lớp được mở khóa khác nhau.
- `_stop_sim`: cấu hình mô hình liên quan đến tập stop/non_stop gốc.

Một số xu hướng có thể đọc được từ summary:

- Với PMA, ASR thường thấp trong các cấu hình fine-tuning đã ghi nhận.
- Với BAMA, FGSM và CW có thể đạt ASR cao ở một số cấu hình, nhưng khi số lớp fine-tune thay đổi thì ASR biến động mạnh.
- Với E-BAMA, FGSM và CW cũng có ASR cao ở nhiều cấu hình, nhưng không ổn định tuyệt đối giữa các mức fine-tuning.
- CAN vẫn là attack yếu trong phần lớn cấu hình, thường có ASR gần 0%.

Khi đưa vào báo cáo, nên trình bày fine-tuning như một nhóm phân tích riêng, vì kết quả phụ thuộc nhiều vào số lớp được fine-tune, không nên gộp chung trực tiếp với feature extraction.

---

## 7. Cấu trúc thư mục và vai trò từng file

```text
.
├── appendix_outputs/
│   ├── attack_summary.csv
│   └── attack_success_summary.png
│
├── figures/figures/
│   ├── CIFAR10/
│   ├── FLOWERS/
│   └── GTSRB/
│
├── fixed_code/
│   └── Bản sao các file code đã chỉnh sửa/cố định
│
├── BAMA.py
├── E-BAMA.py
├── PMA.py
├── attack_common.py
├── backbone_utils.py
├── fea_ext_binary.py
├── fin_tun_binary.py
├── make_datasets.py
├── summarize_results.py
├── tflite_converter.py
├── smartappattack_requirements.txt
├── attack_summary.csv
└── attack_success_summary.png
```

Vai trò cụ thể:

| File/thư mục | Vai trò |
|---|---|
| `make_datasets.py` | Tạo dữ liệu nhị phân `stop` / `non_stop` cho các dataset. |
| `backbone_utils.py` | Khai báo backbone, hàm preprocess và mô hình binary classifier. |
| `fea_ext_binary.py` | Huấn luyện mô hình theo chế độ feature extraction. |
| `fin_tun_binary.py` | Huấn luyện mô hình theo chế độ fine-tuning. |
| `tflite_converter.py` | Chuyển SavedModel sang định dạng `.tflite`. |
| `attack_common.py` | Chứa logic chính để chạy attack, sinh adversarial examples và tính ASR. |
| `BAMA.py` | Wrapper để chạy attack với baseline BAMA. |
| `E-BAMA.py` | Wrapper để chạy attack với baseline E-BAMA. |
| `PMA.py` | Wrapper để chạy attack với baseline PMA. |
| `summarize_results.py` | Quét các file JSON kết quả và tạo summary CSV + biểu đồ. |
| `appendix_outputs/` | Kết quả tổng hợp nên dùng cho báo cáo/phụ lục. |
| `figures/figures/` | Hình loss/accuracy của quá trình huấn luyện. |
| `fixed_code/` | Bản code đã chỉnh sửa, dùng để đối chiếu hoặc thay thế khi cần. |

---

## 8. Các thuật toán attack trong repo

Repo hỗ trợ các attack sau:

| Attack | Ý nghĩa ngắn |
|---|---|
| `FGSM` | Fast Gradient Sign Method, attack một bước dựa trên dấu gradient. |
| `CW` | Carlini & Wagner attack, thường mạnh hơn nhưng tốn thời gian hơn. |
| `CAN` | Clipping-aware additive noise attack trong Foolbox. |
| `PGD` | Projected Gradient Descent, attack nhiều bước, thường mạnh hơn FGSM. |
| `MIFGSM` / `MI-FGSM` | Momentum Iterative FGSM, dùng momentum để tăng khả năng tấn công. |

Trong kết quả hiện tại, các attack có hiệu quả cao nhất là `PGD`, `MI-FGSM`, `CW` và `FGSM` trên BAMA/E-BAMA. `CAN` cho kết quả thấp trong các dòng đã tổng hợp.

---

## 9. Cài đặt môi trường

Cài thư viện theo file requirements:

```bash
pip install -r smartappattack_requirements.txt
```

Các thư viện chính gồm:

- TensorFlow 2.8.0
- Foolbox 3.3.3
- EagerPy 0.30.0
- NumPy
- Matplotlib
- Pillow
- tqdm
- TensorFlow Datasets
- pandas

---

## 10. Chạy lại pipeline cơ bản

### 10.1. Tạo dataset nhị phân

Ví dụ với GTSRB:

```bash
python make_datasets.py --dataset GTSRB --quick
```

Nếu muốn tạo đủ dữ liệu hơn, bỏ `--quick`:

```bash
python make_datasets.py --dataset GTSRB
```

### 10.2. Huấn luyện feature extraction

Ví dụ huấn luyện BAMA với MobileNetV2:

```bash
python fea_ext_binary.py \
  --dataset GTSRB \
  --backbone MobileNetV2 \
  --baseline BAMA \
  --epochs 2
```

Ví dụ huấn luyện E-BAMA:

```bash
python fea_ext_binary.py \
  --dataset GTSRB \
  --backbone MobileNetV2 \
  --baseline E-BAMA \
  --epochs 2
```

### 10.3. Chạy attack

Ví dụ chạy FGSM trên BAMA:

```bash
python BAMA.py \
  --dataset GTSRB \
  --backbone MobileNetV2 \
  --training_mode feature_extraction \
  --model_name MobileNetV2_GTSRB_BAMA_stop_sim \
  --attack_algo FGSM \
  --rounds 3 \
  --num_images 50
```

Ví dụ chạy PGD trên E-BAMA:

```bash
python E-BAMA.py \
  --dataset GTSRB \
  --backbone MobileNetV2 \
  --training_mode feature_extraction \
  --model_name MobileNetV2_GTSRB_E-BAMA_stop_sim \
  --attack_algo PGD \
  --rounds 3 \
  --num_images 50 \
  --steps 10 \
  --random_start
```

### 10.4. Tổng hợp kết quả

Sau khi đã có các file JSON trong `results_json/`, chạy:

```bash
python summarize_results.py
```

Script sẽ tạo:

```text
appendix_outputs/attack_summary.csv
appendix_outputs/attack_success_summary.png
```

---
