import sys
import os
import random
import matplotlib.pyplot as plt

# Thêm đường dẫn thư mục gốc để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import DataLoader

def main():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("Hãy bỏ vài tấm ảnh vào thư mục test/input/")
        return
        
    img_name = random.choice(files)
    img_path = os.path.join(input_dir, img_name)
    
    loader = DataLoader()
    # is_input=True để chạy đủ bộ lọc OpenCV
    X_flat = loader.preprocess_image(img_path, is_input=True)
    
    if X_flat is None:
        return

    # Phục hồi vector 1D (784,) về lại ma trận 2D (28, 28) để hiển thị
    img_28x28 = X_flat.reshape(28, 28)

    # --- VẼ BIỂU ĐỒ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Bước 0: Tiền xử lý ảnh - File: {img_name}', fontsize=16)

    # Hình 1: Hiển thị ảnh 28x28
    cax = ax1.imshow(img_28x28, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Ma trận 28x28 (Giá trị 0.0 - 1.0)')
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

    # Hình 2: Biểu đồ Histogram xem phân bố pixel
    ax2.hist(X_flat, bins=50, color='blue', alpha=0.7)
    ax2.set_title('Phân bố 784 giá trị Pixel')
    ax2.set_xlabel('Giá trị Pixel (0=Đen, 1=Trắng)')
    ax2.set_ylabel('Số lượng Pixel')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()