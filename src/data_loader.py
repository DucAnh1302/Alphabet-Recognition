# src/data_loader.py
import os
import cv2
import numpy as np
from src.config import IMG_SIZE, OUTPUT_PATH # Import thêm OUTPUT_PATH để lưu ảnh debug

class DataLoader:
    def __init__(self):
        pass
    
    # Hàm này mình sinh ra với mục tiêu: Nhận vào đường dẫn 1 bức ảnh, làm sạch nó, đưa về kích thước chuẩn 28 x 28, và biến nó thành mảng 784 con số.
    def preprocess_image(self, image_path, is_input=False):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE biến ảnh màu về dạng ảnh xám, trắng và đen
            if img is None:
                return None
            
            if not is_input: # Cờ is_input dùng để phân biệt: Nếu là ảnh lấy từ tập Data (Kaggle) thì nó đã chuẩn sẵn rồi, chỉ cần cv2.resize về 28x28 và flatten() luôn.
                img_resized = cv2.resize(img, IMG_SIZE)
                return img_resized.flatten() / 255.0
            
            # --- ẢNH INPUT (PAINT/CAMERA) ---
            # Đảm bảo ảnh luôn ở dạng nền trắng chữ đen (Chống lỗi Dark mode Paint)
            if np.mean(img) < 127: # ảnh thuộc điều kiện này là ảnh tối màu mà chữ sáng
                img = 255 - img # đảo ngược lại lấy 255 - đi số liệu đó để lật màu lại
                
            # Xóa đường kẻ ô ly (Làm mờ nhẹ để các nét mỏng màu xanh bay màu)
            blurred = cv2.GaussianBlur(img, (7, 7), 0)
            
            # Trị bóng đổ bằng Adaptive Threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 41, 15  # lật ngược ảnh, biến nền đen 0 thành chữ trắng 255
            )
            
            if thresh is None or not isinstance(thresh, np.ndarray):
                print(f"thresh không hợp lệ cho {image_path}")
                return None
            
            # Lọc rác và Cắt khung chữ
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Lọc các hạt bụi nhỏ
                valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea) # Tìm cái viền to nhất trong ảnh, giả định đó chính là chữ cái cần tìm.
                    x, y, w, h = cv2.boundingRect(largest_contour) # vẽ 1 hình chữ nhật bao gồm nó, x, y là biến tọa độ góc trái bên trên, còn w và h là rộng và dài
                    
                    # nếu khung cắt to bằng cả tờ giấy thì bỏ qua
                    if w * h > 0.8 * img.shape[0] * img.shape[1]: # tờ giấy trắng thì cắt luôn
                        roi = thresh
                    else:
                        roi = thresh[y:y+h, x:x+w] # cắt đúng vùng chữ cái (ROI - Region of interest).
                else:
                    roi = thresh
            else:
                roi = thresh
                w, h = roi.shape[1], roi.shape[0]

            # Làm đậm nét vẽ (Giảm lực lại một chút để chữ không bị dính cục)
            thickness = max(1, int(max(w, h) / 25)) # tính toán độ dày, khung càng to thì tô càng dày
            kernel = np.ones((thickness, thickness), np.uint8) # tạo 1 cái ma trận hình vuông với "cọ vẽ là 1"
            roi = cv2.dilate(roi, kernel, iterations=1) # Hàm dilate (giãn nở). Nó dùng cái "cọ vẽ" kernel quét qua chữ cái, làm cho các nét trắng phình to ra (đậm lên).
            
            # Căn giữa khung bạt 28x28
            side = max(w, h) # w = 100 và h = 200
            pad = int(side * 0.15) # 200 * 0.15 = 30px
            canvas_side = side + 2 * pad # 200 + 2 * 30 = 260px
            canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8) # Tạo một khung bạt hình vuông (canvas) toàn màu đen (số 0). Khung này to hơn chữ cái một chút để tạo lề (pad).
            
            ax = pad + (side - w) // 2 # ax = 30 + (200 - 100) // 2 = 30 + 50 = 80
            ay = pad + (side - h) // 2 # ay = 30 + (200 - 200) // 2 = 30 + 0 = 30
            canvas[ay:ay+h, ax:ax+w] = roi # dán chữ vào giữa khung
            
            img_resized = cv2.resize(canvas, IMG_SIZE, interpolation=cv2.INTER_AREA) # thay đổi kích thước cho đúng 28x28
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
    # Hàm load việc training thôi
#    def load_training_data(self, root_folder):
#        X_data = [] 
#        y_labels = [] 
#        #print(f"Đang tải tối đa {max_per_class} ảnh cho mỗi chữ cái...")
#        print(f"Đang tải dữ liệu...")
#
#        for dirpath, dirnames, filenames in os.walk(root_folder):
#            folder_name = os.path.basename(dirpath)
#            if not filenames:
#                continue
#
#            count = 0 
#            for file in filenames:
#                #if count >= max_per_class:
#                #    break 
#                    
#                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                    img_path = os.path.join(dirpath, file)
#                    processed_img = self.preprocess_image(img_path, is_input=False)
#                    
#                    if processed_img is not None:
#                        X_data.append(processed_img)
#                        y_labels.append(folder_name.lower())
#                        count += 1 
#
#        return np.array(X_data), np.array(y_labels)
#

#    def load_training_data(self, root_folder):
#        X_data = [] 
#        y_labels = [] 
#        print("Đang tải dữ liệu (Chỉ lấy chữ In Hoa A-Z)...")
#
#        # Tập hợp 26 chữ cái in hoa để làm bộ lọc
#        valid_labels = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#
#        for dirpath, dirnames, filenames in os.walk(root_folder):
#            folder_name = os.path.basename(dirpath)
#            if not filenames:
#                continue
#
#            # BỘ LỌC CHÍNH: Bỏ qua nếu tên thư mục không nằm trong danh sách A-Z
#            if folder_name not in valid_labels:
#                continue
#
#            count = 0 
#            for file in filenames:
#                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                    img_path = os.path.join(dirpath, file)
#                    processed_img = self.preprocess_image(img_path, is_input=False)
#                    
#                    if processed_img is not None:
#                        X_data.append(processed_img)
#                        # Giữ nguyên chữ in hoa làm nhãn (không dùng .lower() nữa)
#                        y_labels.append(folder_name) 
#                        count += 1 
#
#        return np.array(X_data), np.array(y_labels)

    def load_training_data(self, root_folder, max_per_class=400):
        X_data = [] 
        y_labels = [] 
        print(f"Đang tải dữ liệu (Tối đa {max_per_class} ảnh/class, chỉ lấy chữ In Hoa A-Z)...")

        # Tập hợp 26 chữ cái in hoa để làm bộ lọc
        valid_labels = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for dirpath, dirnames, filenames in os.walk(root_folder):
            folder_name = os.path.basename(dirpath)
            if not filenames:
                continue

            # BỘ LỌC CHÍNH: Bỏ qua nếu tên thư mục không nằm trong danh sách A-Z
            if folder_name not in valid_labels:
                continue

            count = 0 
            for file in filenames:
                # KÍCH HOẠT LẠI GIỚI HẠN SỐ LƯỢNG ẢNH
                if count >= max_per_class:
                    break 
                    
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(dirpath, file)
                    processed_img = self.preprocess_image(img_path, is_input=False)
                    
                    if processed_img is not None:
                        X_data.append(processed_img)
                        # Giữ nguyên chữ in hoa làm nhãn
                        y_labels.append(folder_name) 
                        count += 1 

        return np.array(X_data), np.array(y_labels)

    # hàm đọc folder: mục tiêu là đọc tất cả ảnh trong thư mục input và thực thi chúng, gán nhãn cho chúng thông qua tên thư mục
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