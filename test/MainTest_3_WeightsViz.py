import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import DataLoader, AlphabetRecognizer
from src.config import MODEL_PATH

def main():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    img_path = os.path.join(input_dir, random.choice(files))
    
    loader = DataLoader()
    X_flat = loader.preprocess_image(img_path, is_input=True)
    
    recognizer = AlphabetRecognizer()
    recognizer.load(MODEL_PATH)
    clf = recognizer.model
    
    # Lấy đáp án dự đoán cuối cùng
    predicted_label = clf.predict([X_flat])[0]
    
    # Tìm index của đáp án này trong mảng classes_
    class_index = np.where(clf.classes_ == predicted_label)[0][0]
    
    # Lấy ra bộ Trọng số (784,) của riêng chữ cái này và phục hồi thành 28x28
    weight_vector = clf.coef_[class_index]
    weight_matrix = weight_vector.reshape(28, 28)

    # --- VẼ BIỂU ĐỒ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Kết quả: Mô hình dự đoán đây là chữ "{predicted_label}"', fontsize=16)

    # Hình 1: Ảnh gốc sau khi tiền xử lý
    ax1.imshow(X_flat.reshape(28, 28), cmap='gray')
    ax1.set_title('Ảnh đầu vào (28x28)')
    ax1.axis('off')

    # Hình 2: "Khuôn mẫu" trọng số của chữ cái đó
    # Màu đỏ (âm) là vùng nét bút KHÔNG NÊN vẽ vào.
    # Màu xanh (dương) là vùng nét bút NÊN CÓ để tạo thành chữ này.
    cax = ax2.imshow(weight_matrix, cmap='RdBu', vmin=-np.max(np.abs(weight_matrix)), vmax=np.max(np.abs(weight_matrix)))
    ax2.set_title(f'Trọng số mô hình đã học cho chữ "{predicted_label}"')
    fig.colorbar(cax, ax=ax2)
    ax2.axis('off')

    plt.show()

if __name__ == "__main__":
    main()