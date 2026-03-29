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
    img_name = random.choice(files)
    img_path = os.path.join(input_dir, img_name)
    
    loader = DataLoader()
    X_flat = loader.preprocess_image(img_path, is_input=True)
    
    recognizer = AlphabetRecognizer()
    recognizer.load(MODEL_PATH)
    clf = recognizer.model
    classes = clf.classes_
    
    # HÀM TÍNH XÁC SUẤT CỦA THƯ VIỆN (Tương đương hàm Softmax)
    probabilities = clf.predict_proba([X_flat])[0] * 100 # Nhân 100 để ra phần trăm
    
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_classes = [classes[i] for i in top_5_indices]
    top_5_probs = [probabilities[i] for i in top_5_indices]

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(top_5_classes, top_5_probs, color='green')
    plt.title(f'Bước 2: Xác suất % (Softmax) - File: {img_name}', fontsize=14)
    plt.xlabel('Chữ cái')
    plt.ylabel('Xác suất (%)')
    plt.ylim(0, 110) # Giới hạn trục Y từ 0 đến 100%
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", va='bottom', ha='center')
        
    plt.show()

if __name__ == "__main__":
    main()