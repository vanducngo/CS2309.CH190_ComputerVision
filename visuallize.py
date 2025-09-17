import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Tải bộ dữ liệu CIFAR-10 gốc ---
# Chúng ta sẽ chỉ lấy tập test để trực quan hóa
transform_original = transforms.Compose([
    transforms.ToTensor()
])

try:
    cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=transform_original)
except Exception as e:
    print(f"Lỗi khi tải CIFAR-10. Vui lòng kiểm tra kết nối mạng. Lỗi: {e}")
    exit()

# Lấy một ảnh ngẫu nhiên từ tập test
random_index = np.random.randint(0, len(cifar10_testset))
original_img_tensor, original_label = cifar10_testset[random_index]
class_names = cifar10_testset.classes
print(f"Đã chọn ảnh ngẫu nhiên index {random_index}, lớp: {class_names[original_label]}")

# Chuyển tensor về dạng ảnh PIL để áp dụng các phép biến đổi nhiễu
original_img_pil = transforms.ToPILImage()(original_img_tensor)


# --- 2. Định nghĩa 15 loại nhiễu (Corruption) ---
# Dựa trên danh sách từ bài báo RoTTA và benchmark CIFAR-10-C
# Lưu ý: Một số phép biến đổi này không có sẵn trong torchvision,
# chúng ta cần tự định nghĩa hoặc sử dụng thư viện chuyên dụng như `imagecorruptions`.
# Để đơn giản, ở đây chúng ta sẽ mô phỏng một vài loại nhiễu có sẵn.
# Để có kết quả chính xác như bài báo, bạn cần cài đặt thư viện `imagecorruptions`.

# Cài đặt thư viện imagecorruptions:
# pip install imagecorruptions
try:
    from imagecorruptions import corrupt
except ImportError:
    print("Thư viện 'imagecorruptions' chưa được cài đặt.")
    print("Vui lòng chạy: pip install imagecorruptions")
    # Sử dụng các phép biến đổi thay thế từ torchvision nếu không có thư viện
    print("Sử dụng các phép biến đổi thay thế đơn giản...")
    use_imagecorruptions = False
else:
    use_imagecorruptions = True


# --- 3. Tạo các ảnh bị nhiễu ---
corrupted_images = []
corruption_names = [
    'motion_blur', 'snow', 'fog', 'shot_noise', 'defocus_blur',
    'contrast', 'zoom_blur', 'brightness', 'frost', 'elastic_transform',
    'glass_blur', 'gaussian_noise', 'pixelate', 'jpeg_compression', 'impulse_noise'
]

# Chuyển ảnh PIL về dạng numpy array để thư viện imagecorruptions xử lý
original_img_numpy = np.array(original_img_pil)
severity_level = 5 # Mức độ nhiễu cao nhất

for name in corruption_names:
    if use_imagecorruptions:
        # Sử dụng thư viện imagecorruptions để tạo nhiễu chính xác
        corrupted_img_numpy = corrupt(original_img_numpy, corruption_name=name, severity=severity_level)
        corrupted_img_pil = Image.fromarray(corrupted_img_numpy)
        corrupted_images.append(corrupted_img_pil)
    else:
        # ---- PHƯƠNG ÁN THAY THẾ (ĐƠN GIẢN HÓA) NẾU KHÔNG CÓ THƯ VIỆN ----
        # Đây chỉ là mô phỏng, không hoàn toàn giống nhiễu trong bài báo
        if 'blur' in name:
            transform_corr = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(3, 5))
        elif 'noise' in name:
            transform_corr = transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.2, 0, 1))
        elif 'contrast' in name or 'brightness' in name:
            transform_corr = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        else: # Các loại nhiễu khác
            transform_corr = transforms.RandomErasing(p=1, scale=(0.2, 0.4))
            
        corrupted_img_tensor = transform_corr(original_img_tensor)
        corrupted_images.append(transforms.ToPILImage()(corrupted_img_tensor))


# --- 4. Trực quan hóa trên Grid 4x4 ---
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle(f'Original vs. 15 Corruptions (Severity 5)\nClass: {class_names[original_label]}', fontsize=20, y=0.95)

# Chuyển đổi axes thành mảng 1D để dễ lặp
axes = axes.ravel()

# Hiển thị ảnh gốc ở vị trí đầu tiên
axes[0].imshow(original_img_pil)
axes[0].set_title("Original", fontsize=12, color='blue')
axes[0].axis('off') # Ẩn các trục tọa độ

# Hiển thị 15 ảnh nhiễu
for i in range(15):
    if i < len(corrupted_images):
        axes[i + 1].imshow(corrupted_images[i])
        # Lấy tên nhiễu và định dạng lại cho đẹp
        title_name = corruption_names[i].replace('_', ' ').title()
        axes[i + 1].set_title(title_name, fontsize=10)
        axes[i + 1].axis('off')

# Tinh chỉnh layout
plt.tight_layout(rect=[0, 0, 1, 0.93])
# Lưu hình ảnh
plt.savefig("cifar10_corruptions_grid.png", dpi=300)
# Hiển thị hình ảnh
plt.show()