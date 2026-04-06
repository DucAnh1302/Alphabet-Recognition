# src/gui.py
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

from src.data_loader import DataLoader
from src.model_logic import AlphabetRecognizer
from src.config import MODEL_PATH, INPUT_PATH

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ in hoa (HOG + Logistic Regression)")
        self.root.geometry("500x650")
        self.root.resizable(False, False)

        # Khởi tạo Lõi AI
        self.loader = DataLoader()
        self.recognizer = AlphabetRecognizer()
        
        # Load model đã train ở Phase 1
        if not self.recognizer.load(MODEL_PATH):
            print("Cảnh báo: Chưa có model, hãy chạy file main.py để train trước!")

        # Kích thước bảng vẽ
        self.canvas_width = 400
        self.canvas_height = 400
        
        # TẠO ẢNH PIL ẨN (Nền trắng 255) ĐỂ LƯU NÉT VẼ
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()
        
        # Biến lưu tọa độ nét vẽ
        self.old_x = None
        self.old_y = None

    def setup_ui(self):
        lbl_title = tk.Label(self.root, text="VẼ MỘT CHỮ CÁI IN HOA (A-Z)", font=("Arial", 16, "bold"))
        lbl_title.pack(pady=10)

        # Bảng trắng Canvas để vẽ
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross", highlightbackground="gray", highlightthickness=2)
        self.canvas.pack(pady=10)

        # Bắt sự kiện chuột (Phase 2)
        self.canvas.bind('<Button-1>', self.start_drawing)       # Click chuột xuống
        self.canvas.bind('<B1-Motion>', self.draw_lines)         # Giữ và rê chuột
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing) # Nhả chuột

        # Khung chứa nút bấm
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        btn_predict = tk.Button(btn_frame, text="Dự đoán", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=12, command=self.predict)
        btn_predict.grid(row=0, column=0, padx=15)

        btn_clear = tk.Button(btn_frame, text="Xóa bảng", font=("Arial", 14), width=12, command=self.clear_canvas)
        btn_clear.grid(row=0, column=1, padx=15)

        # Nhãn hiện kết quả
        self.lbl_result = tk.Label(self.root, text="Kết quả: ...", font=("Arial", 24, "bold"), fg="#D32F2F")
        self.lbl_result.pack(pady=15)

    def start_drawing(self, event):
        self.old_x = event.x
        self.old_y = event.y

    def draw_lines(self, event):
        if self.old_x and self.old_y:
            # 1. Vẽ lên màn hình Tkinter cho người dùng thấy
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=20, fill="black", capstyle=tk.ROUND, smooth=True)
            
            # 2. Vẽ y hệt lên ảnh PIL ẩn để lát đưa cho Model
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=0, width=20, joint="curve")
            
        self.old_x = event.x
        self.old_y = event.y

    def stop_drawing(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        # Xóa Canvas Tkinter
        self.canvas.delete("all")
        # Xóa ảnh PIL (tạo lại tờ giấy trắng mới)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Kết quả: ...")

    # PHASE 3: TÍCH HỢP (INTEGRATION)
    def predict(self):
        if not self.recognizer.is_trained:
            self.lbl_result.config(text="Lỗi: Model chưa học!")
            return

        # 1. Chuyển ảnh PIL thành ma trận Numpy
        img_array = np.array(self.image)
        
        # 2. Lưu tạm ra file để lợi dụng lại hàm preprocess_image
        if not os.path.exists(INPUT_PATH):
            os.makedirs(INPUT_PATH)
        temp_path = os.path.join(INPUT_PATH, "temp_draw.jpg")
        cv2.imwrite(temp_path, img_array)

        # 3. Đưa vào luồng xử lý: Cắt viền, căn giữa, HOG, Logistic Regression
        X_flat = self.loader.preprocess_image(temp_path, is_input=True)
        
        if X_flat is not None:
            # 4. Nhờ "não" dự đoán
            prediction = self.recognizer.predict(np.array([X_flat]))[0]
            probabilities = self.recognizer.predict_proba(np.array([X_flat]))[0] * 100
            max_prob = np.max(probabilities)
            
            # Hiển thị lên UI
            self.lbl_result.config(text=f"Chữ: {prediction} ({max_prob:.1f}%)")
        else:
            self.lbl_result.config(text="Lỗi tiền xử lý ảnh!")