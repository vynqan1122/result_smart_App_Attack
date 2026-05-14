# PHỤ LỤC: Hướng dẫn thực nghiệm SmartAppAttack

## 1. Mục tiêu

Tái lập SmartAppAttack trên ba bộ dữ liệu GTSRB, CIFAR-10 và Oxford Flowers, ba backbone MobileNetV2, InceptionV3, ResNet50V2, hai hướng transfer learning Feature Extraction/Fine-Tuning, ba baseline PMA/BAMA/E-BAMA và ba thuật toán FGSM/C&W/CAN.

## 2. Ý nghĩa thuật toán

SmartAppAttack là tấn công grey-box vào mô hình học sâu chạy trên thiết bị. Vì mô hình TFLite khó dùng trực tiếp để tính gradient, ta huấn luyện một mô hình nhị phân tương tự mô hình nạn nhân rồi sinh ảnh adversarial trên mô hình nhị phân đó. Ảnh adversarial được kiểm tra lại trên mô hình TFLite.

## 3. Các file chính

- `make_datasets.py`: tạo dataset nhị phân target/non-target.
- `fea_ext_binary.py`: huấn luyện binary model bằng Feature Extraction.
- `fin_tun_binary.py`: huấn luyện binary model bằng Fine-Tuning.
- `tflite_converter.py`: chuyển SavedModel sang TFLite.
- `attack_common.py`: chạy attack, lọc ảnh đúng, tính ASR.
- `BAMA.py`, `E-BAMA.py`, `PMA.py`: wrapper baseline.
- `summarize_results.py`: xuất CSV và biểu đồ.

## 4. Chạy thực nghiệm

```bash
./run_smartappattack.sh quick
./run_smartappattack.sh full
```

## 5. Attack Success Rate

```text
ASR = số ảnh target bị TFLite phân loại sai / số ảnh target được attack
```

Chỉ ảnh target được TFLite phân loại đúng trước attack mới được dùng tính ASR.
