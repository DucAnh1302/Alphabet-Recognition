# src/data_loader.py
import os
import cv2
import numpy as np
from src.config import IMG_SIZE, OUTPUT_PATH # Import thêm OUTPUT_PATH để lưu ảnh debug

class DataLoader:
    def __init__(self):
        pass

    def preprocess_image(self, image_path, is_input=False):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            if not is_input:
                img_resized = cv2.resize(img, IMG_SIZE)
                return img_resized.flatten() / 255.0
            
            # --- ẢNH INPUT (PAINT/CAMERA) ---
            # 1. Đảm bảo ảnh luôn ở dạng nền trắng chữ đen (Chống lỗi Dark mode Paint)
            if np.mean(img) < 127:
                img = 255 - img
                
            # 2. Xóa đường kẻ ô ly (Làm mờ nhẹ để các nét mỏng màu xanh bay màu)
            blurred = cv2.GaussianBlur(img, (7, 7), 0)
            
            # 3. Trị bóng đổ bằng Adaptive Threshold (Kỹ thuật xịn nhất của Computer Vision)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 41, 15
            )
            
            # 4. Lọc rác & Cắt khung chữ
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Lọc các hạt bụi nhỏ
                valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Chống lỗi "Cục gạch trắng" (nếu khung cắt to bằng cả tờ giấy thì bỏ qua)
                    if w * h > 0.8 * img.shape[0] * img.shape[1]:
                        roi = thresh
                    else:
                        roi = thresh[y:y+h, x:x+w]
                else:
                    roi = thresh
            else:
                roi = thresh
                w, h = roi.shape[1], roi.shape[0]

            # 5. Làm đậm nét vẽ (Giảm lực lại một chút để chữ không bị dính cục)
            thickness = max(1, int(max(w, h) / 25))
            kernel = np.ones((thickness, thickness), np.uint8)
            roi = cv2.dilate(roi, kernel, iterations=1)
            
            # 6. Căn giữa khung bạt 28x28
            side = max(w, h)
            pad = int(side * 0.15)
            canvas_side = side + 2 * pad
            canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
            
            ax = pad + (side - w) // 2
            ay = pad + (side - h) // 2
            canvas[ay:ay+h, ax:ax+w] = roi
            
            img_resized = cv2.resize(canvas, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img_resized = cv2.dilate(img_resized, np.ones((2,2), np.uint8), iterations=1)

            # Lưu ảnh Debug
            debug_dir = os.path.join(OUTPUT_PATH, "Debug")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            debug_path = os.path.join(debug_dir, f"debug_{os.path.basename(image_path)}")
            cv2.imwrite(debug_path, img_resized)
            
            return img_resized.flatten() / 255.0
            
        except Exception as e:
            print(f"Lỗi tiền xử lý: {e}")
            return None

    #def load_training_data(self, root_folder, max_per_class=3000):
    def load_training_data(self, root_folder):
        X_data = [] 
        y_labels = [] 
        #print(f"Đang tải tối đa {max_per_class} ảnh cho mỗi chữ cái...")
        print(f"Đang tải dữ liệu...")

        for dirpath, dirnames, filenames in os.walk(root_folder):
            folder_name = os.path.basename(dirpath)
            if not filenames:
                continue

            count = 0 
            for file in filenames:
                #if count >= max_per_class:
                #    break 
                    
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(dirpath, file)
                    processed_img = self.preprocess_image(img_path, is_input=False)
                    
                    if processed_img is not None:
                        X_data.append(processed_img)
                        y_labels.append(folder_name.lower())
                        count += 1 

        return np.array(X_data), np.array(y_labels)

    def load_input_folder(self, input_folder):
        images = []
        filenames = []
        files = sorted(os.listdir(input_folder))
        
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(input_folder, file)
                # Đánh dấu đây là ảnh Input để hàm lưu ảnh debug
                processed_img = self.preprocess_image(img_path, is_input=True)
                if processed_img is not None:
                    images.append(processed_img)
                    filenames.append(file)
        
        return np.array(images), filenames