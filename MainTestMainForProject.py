# main.py
import os
from src.config import DATA_PATH, INPUT_PATH, OUTPUT_PATH, MODEL_PATH
from src import DataLoader, AlphabetRecognizer
from datetime import datetime

def main():
    # 1. Khởi tạo các module
    accuracy = 0
    data_loader = DataLoader()
    recognizer = AlphabetRecognizer()

    # Kiểm tra xem có model đã lưu chưa, nếu chưa thì train mới
    if not recognizer.load(MODEL_PATH):
        print("Không tìm thấy model cũ, bắt đầu quá trình Training mới...")
        
        # Load dữ liệu từ folder Data
        if not os.path.exists(DATA_PATH):
            print("Lỗi: Không tìm thấy thư mục Data!")
            return

        # Gọi hàm và truyền số lượng muốn lấy là biến max_per_class (ví dụ 400 ảnh mỗi chữ thì max_per_class = 400)
        X, y = data_loader.load_training_data(DATA_PATH, max_per_class=2500)
        #X, y = data_loader.load_training_data(DATA_PATH)
        
        if len(X) == 0:
            print("Lỗi: Không tìm thấy ảnh nào trong Data.")
            return

        # Train model
        accuracy = recognizer.train(X, y)
        recognizer.save(MODEL_PATH)
    
    # Dự đoán ảnh trong thư mục Input
    print(f"\nĐang đọc dữ liệu từ: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
        print("Đã tạo thư mục Input. Hãy bỏ ảnh vào và chạy lại.")
        return

    X_input, filenames = data_loader.load_input_folder(INPUT_PATH)
    
    if len(X_input) == 0:
        print("Thư mục Input trống. Hãy bỏ file jpg vào.")
        return

    # Thực hiện dự đoán
    predictions = recognizer.predict(X_input)

    # Ghi kết quả ra file Output
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    output_file = os.path.join(OUTPUT_PATH, 'result.txt')
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # Định dạng theo DD/MM/YYYY HH:MM:SS
        now = datetime.now()
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
        # ghi nhận %
        if(accuracy != 0):
            f.write(f"Thời gian ghi nhận train: {formatted_time} \n")
            f.write(f"độ chính xác là: {accuracy:.2f} % \n")
        else:
            print("sử dụng độ chính xác cũ \n")
            f.write(f"Thời gian ghi nhận với độ chính xác cũ: {formatted_time}\n")
            pass
        f.write(f"{'Tên File':<30} | {'Kết quả dự đoán'}\n")
        f.write("-" * 50 + "\n")
        
        for filename, label in zip(filenames, predictions):
            line = f"{filename:<30} | {label}"
            print(line) # In ra màn hình console
            f.write(line + "\n") # Ghi vào file
            
    print(f"\nĐã ghi kết quả vào: {output_file}")

if __name__ == "__main__":
    main()