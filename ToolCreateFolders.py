import os
import string

# Đường dẫn đến thư mục Data (sửa lại nếu bạn để đường dẫn khác)
base_path = "Data"

# Định nghĩa các loại thư mục muốn tạo
structure = {
    "Uppercase": string.ascii_uppercase, # Tự sinh ra 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #"Lowercase": string.ascii_lowercase, # Tự sinh ra 'abcdefghijklmnopqrstuvwxyz'
    #"Numbers": string.digits           # Thêm dòng này nếu bạn muốn tạo 0-9
}

def create_dataset_structure():
    print(f"Đang tạo thư mục tại: {os.path.abspath(base_path)}")
    
    for sub_folder, chars in structure.items():
        # Tạo đường dẫn: Data/Uppercase hoặc Data/Lowercase
        path = os.path.join(base_path, sub_folder)
        
        # Tạo thư mục cha (Uppercase/Lowercase) trước
        os.makedirs(path, exist_ok=True)
        
        # Tạo từng thư mục con A, B, C... hoặc a, b, c...
        for char in chars:
            char_path = os.path.join(path, char)
            try:
                os.makedirs(char_path, exist_ok=True) # exist_ok=True để không lỗi nếu folder đã có
                print(f"Đã tạo: {char_path}")
            except OSError as e:
                print(f"Lỗi tạo {char_path}: {e}")

    print("\nHoàn tất! Kiểm tra lại thư mục Data nhé.")

if __name__ == "__main__":
    create_dataset_structure()