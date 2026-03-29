import pandas as pd
import numpy as np
import cv2
import os

# --- CẤU HÌNH ---
# Đường dẫn đến file csv bạn tải từ Kaggle (Sửa lại tên file cho đúng)
CSV_PATH = "emnist-byclass-train.csv" 
OUTPUT_DIR = "Data"
IMG_SIZE = 64  # Kích thước ảnh bạn muốn save

# EMNIST Mapping (Theo tài liệu chuẩn của EMNIST)
# 0-9: Số (Ta bỏ qua hoặc lưu nếu muốn)
# 10-35: Chữ hoa (A-Z)
# 36-61: Chữ thường (a-z)
def get_label_folder(label_code):
    if 0 <= label_code <= 9:
        return None, None # Bỏ qua số (hoặc bạn tự thêm code xử lý số)
    elif 10 <= label_code <= 35:
        char = chr(label_code - 10 + ord('A'))
        return "Uppercase", char
    elif 36 <= label_code <= 61:
        char = chr(label_code - 36 + ord('a'))
        return "Lowercase", char
    return None, None

def convert_csv_to_images():
    print("Đang đọc file CSV (sẽ hơi lâu vì file nặng)...")
    
    # Đọc file CSV (Dùng chunksize để không bị tràn RAM nếu file quá lớn)
    # File EMNIST không có header, cột 0 là nhãn
    chunk_size = 10000 
    reader = pd.read_csv(CSV_PATH, header=None, chunksize=chunk_size)
    
    count = 0
    
    for chunk in reader:
        # Duyệt qua từng dòng trong lô dữ liệu
        for index, row in chunk.iterrows():
            label_code = row[0]
            pixels = row[1:].values
            
            # Xác định chữ cái và thư mục
            folder_type, char_name = get_label_folder(label_code)
            
            if folder_type is None:
                continue # Bỏ qua nếu không phải chữ cái
                
            # Tạo đường dẫn
            save_dir = os.path.join(OUTPUT_DIR, folder_type, char_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Xử lý ảnh
            # Dữ liệu gốc là 28x28
            img_array = pixels.reshape(28, 28).astype('uint8')
            
            # LƯU Ý QUAN TRỌNG: Dữ liệu EMNIST gốc bị xoay ngang và lật
            # Phải xoay lại cho đúng chiều người đọc
            img_array = np.transpose(img_array)
            
            # Resize về 64x64
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            # Lưu file
            filename = f"kaggle_{count}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), img_resized)
            
            count += 1
            if count % 1000 == 0:
                print(f"Đã xử lý {count} ảnh...", end='\r')
                
            # (Tùy chọn) Giới hạn số lượng để test nhanh
            # if count >= 5000: break 
        
    print(f"\n Hoàn tất! Đã xả {count} ảnh vào thư mục Data.")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy tải từ Kaggle về trước.")
    else:
        convert_csv_to_images()