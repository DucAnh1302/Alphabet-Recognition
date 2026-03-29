# src/config.py
import os

# Kích thước ảnh chuẩn hóa (Logistic Regression cần input cố định)
IMG_SIZE = (28, 28) 

# Đường dẫn (Tự động lấy theo thư mục hiện tại)
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, 'Data')
INPUT_PATH = os.path.join(BASE_DIR, 'Input')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Output')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl') # Để lưu model đã train