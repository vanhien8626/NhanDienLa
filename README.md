# NhanDienLa — Phân loại bệnh lá cây bằng CNN và Grey Wolf Optimizer

Dự án độc lập thực hiện quy trình trong đề bài: dữ liệu ảnh → CNN/transfer learning → GWO tối ưu siêu tham số → so sánh baseline → đánh giá và Grad-CAM. Không liên quan tới kho hỏi–đáp y tế trước đó.

## Ảnh mẫu và chạy demo ngay

Sau khi cài thư viện ở bước 1, chạy một lệnh để tạo 40 ảnh lá minh họa, huấn luyện CNN/GWO và dự đoán một ảnh:

```bash
python demo.py
```

Ảnh mẫu **được vẽ tổng hợp**, không phải ảnh lá bệnh thật và không dùng để công bố độ chính xác nhận diện bệnh. Hai lớp có tên `synthetic_plain` / `synthetic_spotted`.

| Lá minh họa không đốm | Lá minh họa có đốm |
|---|---|
| ![Ảnh tổng hợp không đốm](docs/demo/plain.png) | ![Ảnh tổng hợp có đốm](docs/demo/spotted.png) |

Demo lưu ảnh đầu vào vào `runs/demo/images/`, mô hình và biểu đồ vào `runs/demo/results/`, ảnh Grad-CAM của lần dự đoán vào `runs/demo/prediction_gradcam.png`. Chạy lại bằng thư mục mới, ví dụ `python demo.py --output runs/demo2`.

Dự đoán lại bằng ảnh mẫu có sẵn sau khi chạy demo:

```bash
python predict.py --checkpoint runs/demo/results/gwo_42.pt --image docs/demo/spotted.png
```

Ảnh Grad-CAM được tạo từ lần chạy demo thực tế (không phải kết quả trên PlantVillage):

![Grad-CAM trên ảnh tổng hợp](docs/demo/gradcam.png)

Đường hội tụ GWO trong lần chạy demo:

![Đường hội tụ trên dữ liệu tổng hợp](docs/demo/gwo.png)

Để dùng ảnh lá thật, huấn luyện trên bộ dữ liệu thật ở bước 2–3, sau đó chạy `predict.py --image DUONG_DAN_ANH` với checkpoint tương ứng. Checkpoint demo chỉ học phân biệt hình tổng hợp.

## 1. Cài môi trường

Khuyến nghị Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell: kích hoạt bằng `.venv\Scripts\Activate.ps1`.

Nếu chỉ dùng CPU, cài torch/torchvision CPU trước để tránh tải các gói CUDA lớn:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Trong môi trường hiện tại đã có interpreter `/workspace/.venvs/plant-disease/bin/python` dùng CPU.

## 2. Chuẩn bị dữ liệu

Tải PlantVillage hoặc bộ ảnh tương đương rồi trỏ tới thư mục **chứa trực tiếp các thư mục lớp**:

```text
leaf_data/
├── Tomato___healthy/
│   ├── 001.jpg
│   └── ...
├── Tomato___Early_blight/
│   ├── 001.jpg
│   └── ...
└── ...
```

Không trỏ vào thư mục cha chứa nhiều phiên bản color/grayscale/segmented, hoặc thư mục đã có train/val/test. Cần ít nhất 2 lớp, mỗi lớp ít nhất 5 ảnh khác nhau.

- Chia riêng theo lớp, xấp xỉ 70% train / 15% validation / 15% test; làm tròn và giữ ít nhất một ảnh mỗi lớp trong validation/test.
- Loại ảnh trùng pixel **trước khi chia** và báo số trùng trong `split.json`; cùng ảnh nhưng khác nhãn sẽ báo lỗi.
- Chỉ tập train có flip, xoay và color jitter. Validation/test dùng phép biến đổi cố định.
- Ảnh gần trùng hoặc nhiều ảnh của cùng cây không được tự phát hiện. Với dữ liệu thực địa cần chia theo cây/nguồn để tránh rò rỉ; cách chia hiện tại là baseline theo ảnh.

## 3. Chạy thử quy trình trên dữ liệu của bạn

```bash
python train.py --data leaf_data --output runs/quick \
  --epochs 2 --search-epochs 1 --wolves 3 --iterations 2 --repeats 1
```

CNN tự xây dựng không cần tải trọng số. Để chạy thí nghiệm dài hơn:

```bash
python train.py --data leaf_data --output runs/cnn \
  --epochs 15 --search-epochs 3 --wolves 5 --iterations 5 --repeats 3
```

ResNet18 transfer learning (lần đầu cần tải trọng số ImageNet từ PyTorch):

```bash
python train.py --data leaf_data --output runs/resnet18 \
  --architecture resnet18 --image-size 224 --device cuda \
  --epochs 15 --search-epochs 3 --wolves 5 --iterations 5 --repeats 3
```

Không có GPU thì dùng `--device cpu`. Dùng `--threads` để giới hạn luồng CPU (mặc định 2). Có thể đặt `TORCH_HOME` để chọn chỗ lưu cache trọng số. `--no-pretrained` chạy ResNet18 ngẫu nhiên để kiểm tra ngoại tuyến, **không còn là transfer learning**.

Chọn thư mục `--output` mới mỗi lần; chương trình từ chối ghi đè thí nghiệm cũ.

## 4. GWO tối ưu những gì?

| Thành phần | Không gian tìm kiếm |
|---|---|
| Learning rate | log-uniform từ 1e-4 đến 1e-2 |
| Batch size | 16, 32, 64 |
| Dropout | 0 đến 0.6 |
| Optimizer | AdamW hoặc SGD momentum 0.9 |
| Số nhóm đóng băng | 0–4 với ResNet18 pretrained; cố định 0 nếu huấn luyện ngẫu nhiên |
| Weight decay | log-uniform từ 1e-6 đến 1e-2 |

`frozen_blocks=1` đóng băng stem; 2 thêm layer1, 3 thêm layer2, 4 thêm layer3; layer4 và đầu phân loại vẫn học. BatchNorm trong nhóm đóng băng được giữ ở eval. CNN có 3 khối Conv–BN–ReLU–Pool và đầu phân loại.

Fitness cần **tối thiểu hóa**:

```text
fitness = 1 - validation_macro_F1
          + complexity_weight × trainable_parameters / total_parameters
```

`--complexity-weight` mặc định 0.02. Đóng băng chỉ giảm số tham số cập nhật, không giảm kích thước hay phép tính suy luận. Với CNN không đóng băng, tỷ lệ tham số là hằng số nên thành phần phạt không thay đổi thứ hạng. Thời gian huấn luyện và tìm kiếm được báo riêng; fitness không trực tiếp tối ưu thời gian.

Mỗi cấu hình dùng cùng seed huấn luyện trong một lần tìm kiếm. Cấu hình trùng được tái sử dụng fitness. Ba sói tốt nhất cập nhật quần thể; lưu nghiệm tốt nhất đã gặp để không mất kết quả.

- Chỉ validation dùng cho fitness và chọn epoch; test chỉ được đánh giá sau khi đã chọn cấu hình.
- Baseline và GWO cuối cùng có cùng ngân sách `--epochs`; GWO phải trả thêm chi phí tìm kiếm.
- Mỗi repeat chạy lại GWO với seed khác, rồi huấn luyện baseline/GWO. Cùng một cách chia dữ liệu giữa các repeat; độ lệch chuẩn phản ánh seed huấn luyện/tìm kiếm, không phải nhiều cách chia dữ liệu.
- Mặc định có tối đa 5×5×3 = 75 lượt đánh giá ứng viên, mỗi lượt 3 epoch, cộng huấn luyện cuối. Hãy dùng cấu hình quick trước khi chạy bộ lớn.
- GWO không đảm bảo tốt hơn baseline; search epoch ít có thể đánh giá sai cấu hình tốt. Cần báo kết quả thực nghiệm thay vì giả định độ chính xác.

## 5. Các tệp kết quả

Trong `--output`:

- `settings.json`, `split.json`: tham số chạy, phiên bản torch, nhãn và ảnh từng tập.
- `trials_SEED.json`: fitness, tham số, thời gian và lịch sử validation của các cấu hình.
- `gwo_SEED.json/.png`: đường hội tụ fitness validation.
- `baseline_SEED.pt`, `gwo_SEED.pt`: checkpoint epoch có validation macro-F1 tốt nhất.
- `results.json`: accuracy, macro-F1, confusion matrix test, tổng/tham số học được, thời gian train/test/search từng repeat.
- `summary.json`: trung bình và độ lệch chuẩn quần thể (ddof=0) accuracy/macro-F1 qua các repeat.
- `*_confusion.png`, `*_learning.png`, `*_gradcam.png`: biểu đồ và Grad-CAM của ảnh test đầu tiên. Grad-CAM giải thích lớp dự đoán, không chứng minh mô hình chẩn đoán đúng.

## 6. Dự đoán ảnh mới

```bash
python predict.py --checkpoint runs/quick/gwo_42.pt \
  --image anh_la.jpg --gradcam gradcam_anh_la.png
```

Trả về tối đa 3 lớp có điểm softmax cao nhất. Điểm softmax chưa hiệu chỉnh không phải xác suất chẩn đoán đáng tin cậy. Tên lớp lấy từ checkpoint; không viết sẵn đáp án. Dự đoán chỉ nhận ảnh, không cần thư mục dữ liệu train.

## 7. Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Các kiểm thử gồm GWO tái lập, chia tập/loại trùng, đóng băng BatchNorm, Grad-CAM và một lượt train → tìm kiếm GWO → lưu checkpoint → nạp/dự đoán trên ảnh tổng hợp. Kết quả trên ảnh tổng hợp chỉ kiểm chứng phần mềm, **không phải kết quả PlantVillage**. Dự án chưa kèm bộ PlantVillage và chưa có chỉ số huấn luyện trên dữ liệu thật. Đã chạy đạt 4 kiểm thử; ngoài ra đã tải trọng số ResNet18 ImageNet từ nguồn chính thức và kiểm chứng forward/backward, đóng băng stem và Grad-CAM.

## Tài liệu đề bài

- https://link.springer.com/article/10.1186/s40537-023-00863-9
- https://arxiv.org/abs/2310.16273

Các liên kết được giữ theo đề bài; triển khai này là một baseline độc lập, không tuyên bố tái lập nguyên xi kết quả các bài báo.
