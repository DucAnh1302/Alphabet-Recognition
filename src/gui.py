# src/gui.py
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import math
import gc

# Nhúng Matplotlib vào Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.data_loader import DataLoader
from src.model_logic import AlphabetRecognizer
from src.config import MODEL_PATH, INPUT_PATH

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XAI Dashboard: Nhận diện chữ in hoa (HOG + Logistic Regression)")
        self.root.geometry("1300x750") # Mở rộng bề ngang để chứa Dashboard
        self.root.configure(bg="#f0f2f5")
        self._is_closing = False

        # Khởi tạo Lõi AI
        self.loader = DataLoader()
        self.recognizer = AlphabetRecognizer()
        
        if not self.recognizer.load(MODEL_PATH):
            print("Cảnh báo: Chưa có model, hãy chạy file main.py để train trước!")

        self.canvas_width = 350
        self.canvas_height = 350
        
        # Ảnh PIL ẩn
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.old_x = None
        self.old_y = None

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # --- KHUNG BÊN TRÁI: BẢNG VẼ ---
        left_frame = tk.Frame(self.root, bg="#f0f2f5", width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        lbl_title = tk.Label(left_frame, text="✏️ VẼ CHỮ IN HOA (A-Z)", font=("Segoe UI", 16, "bold"), bg="#f0f2f5", fg="#333")
        lbl_title.pack(pady=10)

        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross", highlightbackground="#bbb", highlightthickness=2)
        self.canvas.pack(pady=10)

        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_lines)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        btn_frame = tk.Frame(left_frame, bg="#f0f2f5")
        btn_frame.pack(pady=15)

        btn_predict = tk.Button(btn_frame, text="🧠 Phân tích AI", font=("Segoe UI", 13, "bold"), bg="#1976D2", fg="white", width=12, relief="flat", command=self.predict)
        btn_predict.grid(row=0, column=0, padx=10)

        btn_clear = tk.Button(btn_frame, text="🗑️ Xóa bảng", font=("Segoe UI", 13), width=12, relief="flat", command=self.clear_canvas)
        btn_clear.grid(row=0, column=1, padx=10)

        self.lbl_result = tk.Label(left_frame, text="Kết quả: ...", font=("Segoe UI", 26, "bold"), bg="#f0f2f5", fg="#D32F2F")
        self.lbl_result.pack(pady=20)

        # --- KHUNG BÊN PHẢI: DASHBOARD XAI CỦA MATPLOTLIB ---
        right_frame = tk.Frame(self.root, bg="white", highlightbackground="#ddd", highlightthickness=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        lbl_dash = tk.Label(right_frame, text="📊 QUÁ TRÌNH XỬ LÝ (EXPLAINABLE AI)", font=("Segoe UI", 16, "bold"), bg="white", fg="#333")
        lbl_dash.pack(pady=5)

        # Tạo Figure chứa 6 ô (2 hàng x 3 cột)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Nhúng plt vào Tkinter
        self.canvas_mat = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_mat.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.clear_dashboard()

    # CÁC HÀM VẼ TKINTER (Giữ nguyên)
    def start_drawing(self, event):
        self.old_x, self.old_y = event.x, event.y

    def draw_lines(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=22, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], fill=0, width=22, joint="curve")
        self.old_x, self.old_y = event.x, event.y

    def stop_drawing(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.image is not None:
            self.image.close()
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Kết quả: ...")
        self.clear_dashboard()

    def clear_dashboard(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
            ax.text(0.5, 0.5, "Chờ dữ liệu...", ha='center', va='center', color='gray')
        self.canvas_mat.draw()

    # --- LÕI TÍCH HỢP XAI ---
    def predict(self):
        if not self.recognizer.is_trained:
            self.lbl_result.config(text="Lỗi: Model chưa học!")
            return

        img_array = np.array(self.image)
        if not os.path.exists(INPUT_PATH):
            os.makedirs(INPUT_PATH)
        temp_path = os.path.join(INPUT_PATH, "temp_draw.jpg")
        cv2.imwrite(temp_path, img_array)

        X_flat = self.loader.preprocess_image(temp_path, is_input=True)
        
        if X_flat is None:
            self.lbl_result.config(text="Lỗi tiền xử lý!")
            return

        # 1. Trích xuất đặc trưng và tính toán toán học
        clf = self.recognizer.model
        classes = clf.classes_
        
        X_hog = self.recognizer.transform_features(np.array([X_flat]))[0]
        
        # Tự tính Z: Z = W*X + b
        Z_scores = np.dot(clf.coef_, X_hog) + clf.intercept_
        
        # Softmax tự động từ thư viện
        probabilities = self.recognizer.predict_proba(np.array([X_flat]))[0] * 100
        
        prediction = classes[np.argmax(probabilities)]
        max_prob = np.max(probabilities)
        self.lbl_result.config(text=f"Chữ dự đoán: {prediction} ({max_prob:.1f}%)")

        # ==============================================================
        # 2. CẬP NHẬT 6 BƯỚC LÊN DASHBOARD MATPLOTLIB
        # ==============================================================
        for ax in self.axes.flat:
            ax.clear()

        # [Ô 1]: Input (Ảnh gốc thu nhỏ)
        ax1 = self.axes[0, 0]
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title("1. Nét vẽ gốc (Canvas)", fontsize=10, fontweight='bold')
        ax1.axis('off')

        # [Ô 2]: Preprocessing (Ma trận 28x28)
        ax2 = self.axes[0, 1]
        ax2.imshow(X_flat.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        ax2.set_title("2. Tiền xử lý (28x28 Pixel)", fontsize=10, fontweight='bold')
        ax2.axis('off')

        # [Ô 3]: HOG Vector
        ax3 = self.axes[0, 2]
        ax3.plot(X_hog, color='purple', linewidth=1)
        ax3.fill_between(range(len(X_hog)), X_hog, color='purple', alpha=0.3)
        ax3.set_title(f"3. Đặc trưng HOG ({len(X_hog)} chiều)", fontsize=10, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Lấy Top 5 cho Z-score và Xác suất
        top_5_idx = np.argsort(Z_scores)[-5:][::-1]
        top_5_classes = [classes[i] for i in top_5_idx]
        top_5_Z = [Z_scores[i] for i in top_5_idx]
        top_5_prob = [probabilities[i] for i in top_5_idx]

        # [Ô 4]: Linear Z-Scores
        ax4 = self.axes[1, 0]
        bars4 = ax4.bar(top_5_classes, top_5_Z, color='orange')
        ax4.set_title("4. Điểm Z (Linear Score)", fontsize=10, fontweight='bold')
        
        # Vẽ thêm hộp chứa Công thức Toán học nổi bên trong biểu đồ
        ax4.text(0.5, 0.85, r'$Z_k = \mathbf{w}_k^T \mathbf{x}_{hog} + b_k$', 
                 transform=ax4.transAxes, fontsize=11, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF8E1', edgecolor='orange', alpha=0.9))

        # Nới rộng trục Y thêm 40% để cột không đâm vào công thức
        ax4.margins(y=0.4) 

        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), va='bottom', ha='center', fontsize=8)

        # [Ô 5]: Softmax Probabilities
        ax5 = self.axes[1, 1]
        bars5 = ax5.bar(top_5_classes, top_5_prob, color='green')
        ax5.set_title("5. Xác suất P (Softmax %)", fontsize=10, fontweight='bold')
        
        # Vẽ thêm hộp chứa Công thức Softmax nổi
        ax5.text(0.5, 0.85, r'$P_k = \frac{e^{Z_k}}{\sum_{j=1}^{26} e^{Z_j}}$', 
                 transform=ax5.transAxes, fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F5E9', edgecolor='green', alpha=0.9))

        # Nới trần trục Y lên 135 (thay vì 110) để có không gian trống cho hộp công thức
        ax5.set_ylim(0, 135) 
        
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, yval+2, f"{yval:.1f}%", va='bottom', ha='center', fontsize=8)

        # [Ô 6]: Trực quan hóa Trọng số W của chữ cái thắng cuộc
        ax6 = self.axes[1, 2]
        win_idx = np.where(classes == prediction)[0][0]
        weight_vector = clf.coef_[win_idx]
        side = int(math.ceil(math.sqrt(len(weight_vector))))
        padded = np.zeros(side * side)
        padded[:len(weight_vector)] = weight_vector
        weight_matrix = padded.reshape(side, side)
        
        ax6.imshow(weight_matrix, cmap='RdBu')
        ax6.set_title(f"6. Khuôn mẫu Trọng số '{prediction}'", fontsize=10, fontweight='bold')
        ax6.axis('off')

        self.fig.canvas.draw()

    def shutdown_resources(self):
        if self._is_closing:
            return
        self._is_closing = True

        # Xoa file input tam de tranh rac du lieu.
        temp_path = os.path.join(INPUT_PATH, "temp_draw.jpg")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

        try:
            if hasattr(self, "canvas_mat") and self.canvas_mat is not None:
                widget = self.canvas_mat.get_tk_widget()
                if widget.winfo_exists():
                    widget.destroy()
        except tk.TclError:
            pass

        if hasattr(self, "fig") and self.fig is not None:
            self.fig.clear()
            plt.close(self.fig)

        if hasattr(self, "image") and self.image is not None:
            self.image.close()

        self.draw = None
        self.image = None
        self.loader = None
        self.recognizer = None
        self.axes = None
        self.canvas_mat = None
        self.fig = None
        gc.collect()

    def on_close(self):
        self.shutdown_resources()
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass