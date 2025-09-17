# Nâng cấp RoTTA với APQ-Mem: For Robust Test-Time Adaptation in Dynamic Scenarios

Đây là mã nguồn cho dự án chuyên đề **Nâng cấp RoTTA với APQ-Mem**, thuộc môn học CS2309.CH190: Chuyên đề nghiên cứu và ứng dụng về Thị giác máy tính. Dự án này xây dựng dựa trên công trình state-of-the-art **RoTTA (CVPR 2023)** và đề xuất một kiến trúc bộ nhớ đệm (Memory Bank) nâng cao nhằm cải thiện cả về hiệu suất tính toán và độ chính xác thích ứng.

## 1. Giới thiệu

Test-Time Adaptation (TTA) là một hướng nghiên cứu quan trọng, cho phép các mô hình học sâu tự thích ứng với sự thay đổi phân phối dữ liệu (domain shift) trong thực tế mà không cần huấn luyện lại. **RoTTA** là một phương pháp hàng đầu trong lĩnh vực này, giải quyết thành công bài toán TTA trong các kịch bản thực tế (PTTA) nơi dữ liệu thay đổi liên tục và có tính tương quan.

Mặc dù RoTTA rất hiệu quả, chúng tôi nhận thấy thành phần cốt lõi của nó là Memory Bank **CSTU** vẫn còn các hạn chế cố hữu về:
-   **Hiệu suất tính toán:** Thao tác tìm kiếm tuyến tính `O(K)` trở thành nút thắt cổ chai khi mở rộng quy mô bộ nhớ.
-   **Khả năng thích ứng:** Cơ chế lão hóa cứng nhắc, phản ứng chậm với các thay đổi đột ngột của môi trường.
-   **Logic lựa chọn:** Quy tắc cứng nhắc có thể dẫn đến các quyết định chưa tối ưu và nguy cơ các mẫu hiếm tồn tại quá lâu trong bộ nhớ.

Do đó, trong dự án này, chúng tôi đề xuất **APQ-Mem (Adaptive Priority-Queue Memory)**, một kiến trúc bộ nhớ đệm nâng cao được thiết kế để giải quyết triệt để các vấn đề trên. Mục tiêu của dự án là tạo ra một phiên bản RoTTA nhanh hơn, thông minh hơn và mạnh mẽ hơn, sẵn sàng hơn cho các ứng dụng thực tế.

## 2. Đóng góp chính

Dự án này mang lại những đóng góp cụ thể sau:

-   **Tối ưu hóa triệt để hiệu suất:** Bằng cách thay thế cấu trúc dữ liệu `List` bằng **Priority Queue (`heapq`)**, chúng tôi đã giảm độ phức tạp của các thao tác tìm và xóa mẫu từ `O(K)` xuống `O(log K)`, giúp giảm hơn 40% tổng thời gian thực thi và mở khóa khả năng sử dụng bộ nhớ lớn hơn.

-   **Giới thiệu cơ chế Thích ứng Thông minh:** Chúng tôi đề xuất cơ chế **Adaptive Aging**, cho phép bộ nhớ tự động điều chỉnh tốc độ lão hóa của các mẫu dựa trên mức độ thay đổi của môi trường, được định lượng bằng `drift signal` tính từ entropy.

-   **Phát triển Logic Cân bằng Linh hoạt:** Chúng tôi phát triển một **Aware Heuristic Score** với cơ chế phạt mềm dựa trên hàm `softmax`, thay thế cho quy tắc cứng của CSTU. Điều này giúp tăng cường khả năng cân bằng lớp một cách linh hoạt và giải quyết rủi ro các mẫu hiếm bị bất tử trong bộ nhớ.

-   **Thực nghiệm và Phân tích Toàn diện:** Chúng tôi đã tiến hành các thí nghiệm ablation study chi tiết trên **CIFAR-10-C** và **CIFAR-100-C**, chứng minh rằng các cải tiến của chúng tôi không chỉ vượt trội về tốc độ mà còn cải thiện đáng kể về Tỷ lệ lỗi so với phương pháp SOTA.

## 3. Nhóm nghiên cứu

| Họ và Tên | MSSV |
| :--- | :--- |
| Văn Đức Ngọ | 240101020 |
| Phạm Thăng Long | 240101016 |
| Nguyễn Hoàng Hải | 240101008 |

**Giảng viên hướng dẫn:** TS. Nguyễn Vinh Tiệp

## 4. Cài đặt

### Yêu cầu
- Python 3.9
- PyTorch
- Các thư viện được liệt kê trong `requirements.txt`

### Cài đặt môi trường
Bạn có thể tạo một môi trường conda mới và cài đặt các gói cần thiết bằng các lệnh sau:

```bash
# Tạo và kích hoạt môi trường conda mới
conda create -n rotta python=3.9.0
conda activate rotta

# Cài đặt pip và các gói phụ thuộc cơ bản
conda install -y ipython pip

# Cài đặt các gói cần thiết từ file requirements.txt
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu
Mã nguồn có thể tự động tải về bộ dữ liệu CIFAR-10-C và CIFAR-100-C trong lần chạy đầu tiên, tuy nhiên quá trình này có thể rất chậm và không ổn định. Chúng tôi khuyến khích bạn tải về thủ công và tạo symbolic link.

1.  Tải về CIFAR-10-C và CIFAR-100-C.
2.  Tạo symbolic link từ thư mục chứa dữ liệu đến thư mục `datasets` của project:

```bash
# Thay thế path_to_... bằng đường dẫn thực tế trên máy của bạn
ln -s /path_to_your_cifar10_c/ datasets/CIFAR-10-C
ln -s /path_to_your_cifar100_c/ datasets/CIFAR-100-C
```

## 5. Chạy thực nghiệm

Việc chuyển đổi giữa các phiên bản (RoTTA gốc, APQ-Mem, và các bước ablation study) được điều khiển thông qua các cờ (flags) trong file cấu hình `configs/adapter/rotta.yaml`.

### Chạy RoTTA Gốc (Tái hiện kết quả)

Để tái hiện kết quả của RoTTA gốc, hãy đảm bảo các cờ sau được thiết lập trong `configs/adapter/rotta.yaml`:
- `IS_USE_IMPROVE: False`

Sau đó, chạy lệnh:

**CIFAR-10:**
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      OUTPUT_DIR RoTTA_Goc/cifar10
```
**CIFAR-100:**
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar100.yaml \
      OUTPUT_DIR RoTTA_Goc/cifar100
```

### Chạy RoTTA-APQ (Phiên bản cải tiến)

Để chạy phiên bản cải tiến cuối cùng của chúng tôi (Full APQ-Mem), hãy thiết lập các cờ sau trong `configs/adapter/rotta.yaml`:
- `IS_USE_IMPROVE: True`
- `USE_QUEUE: True`
- `USE_AGING_PER_BATCH: True`
- `USE_ADAPTIVE_AGING: True`
- `USE_AWARE_SCORE: True`

Sau đó, chạy các lệnh tương tự:

**CIFAR-10:**
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar10.yaml \
      OUTPUT_DIR RoTTA_APQ/cifar10
```
**CIFAR-100:**
```bash
python ptta.py \
      -acfg configs/adapter/rotta.yaml \
      -dcfg configs/dataset/cifar100.yaml \
      OUTPUT_DIR RoTTA_APQ/cifar100
```

**Lưu ý:** Các siêu tham số mới cho APQ-Mem như `BASE_AGING_SPEED`, `AGE_FACTOR`, `LAMBDA_D` cũng được định nghĩa trong file `configs/adapter/rotta.yaml` và có thể được tinh chỉnh để thực hiện các nghiên cứu sâu hơn.