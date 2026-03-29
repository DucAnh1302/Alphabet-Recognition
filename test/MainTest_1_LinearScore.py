import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import DataLoader, AlphabetRecognizer
from src.config import MODEL_PATH

def main():
    # 1. Bốc ảnh ngẫu nhiên và tiền xử lý
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_path = os.path.join(input_dir, random.choice(files))
    
    loader = DataLoader()
    X_flat = loader.preprocess_image(img_path, is_input=True)
    
    # 2. Load model
    recognizer = AlphabetRecognizer()
    if not recognizer.load(MODEL_PATH):
        print("Lỗi: Không tìm thấy file saved_model.pkl")
        return
        
    clf = recognizer.model
    classes = clf.classes_ # Danh sách các nhãn ('a', 'b', 'c'...)
    
    # Lấy W (trọng số) và b (bias) từ model scikit-learn
    W = clf.coef_ # Ma trận kích thước (số_class, 784)
    b = clf.intercept_  # Vector kích thước (số_class,)
    
    # TỰ TÍNH CÔNG THỨC TOÁN HỌC: Z = W * X + b
    Z_scores = np.dot(W, X_flat) + b
    
    # Lấy top 5 chữ cái có điểm Z cao nhất
    top_5_indices = np.argsort(Z_scores)[-5:][::-1]
    top_5_classes = [classes[i] for i in top_5_indices]
    top_5_scores = [Z_scores[i] for i in top_5_indices]

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(top_5_classes, top_5_scores, color='orange')
    plt.title('Bước 1: Điểm số Tuyến tính Z (Chưa phải xác suất %)', fontsize=14)
    plt.xlabel('Chữ cái dự đoán')
    plt.ylabel('Điểm thô (Z-score)')
    
    # Hiển thị số trên cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')
        
    plt.show()

if __name__ == "__main__":
    main()