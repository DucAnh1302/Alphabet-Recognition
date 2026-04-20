# src/model_logic.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import cv2
import numpy as np

class AlphabetRecognizer:
    def __init__(self):
        # HOG descriptor giúp logistic regression học đường nét tốt hơn nhiều so với pixel thô.
        self.hog = cv2.HOGDescriptor(
            (28, 28),  # winSize: kích thước cửa sổ
            (14, 14),  # blockSize: kích thước ô cơ sở
            (7, 7),    # blockStride: kích thước chuẩn hóa
            (7, 7),    # cellSize: bước trượt
            9          # nbins
        )

        self.model = LogisticRegression(
            max_iter=3500,
            solver='lbfgs',
            n_jobs=-1,
            C=6.0,
            class_weight=None
        )
        self.is_trained = False # cờ để kiểm tra mô hình train chưa thôi

    def _extract_hog_features(self, X):
        if X.ndim != 2:
            raise ValueError("X phải có dạng (n_samples, n_features)")

        # Nếu đã là vector HOG thì trả về luôn.
        if X.shape[1] == self.hog.getDescriptorSize():
            return X.astype(np.float32)

        if X.shape[1] != 28 * 28:
            raise ValueError("Số chiều đầu vào không hợp lệ. Cần 784 (pixel) hoặc kích thước HOG.")

        imgs = (X.reshape(-1, 28, 28) * 255.0).clip(0, 255).astype(np.uint8)
        hog_features = np.array([self.hog.compute(img).reshape(-1) for img in imgs], dtype=np.float32)
        return hog_features

    def transform_features(self, X):
        return self._extract_hog_features(X)

    # nhập số lượng ảnh x và đáp án y để học
    def train(self, X, y):
        print(f"Bắt đầu training với {len(X)} mẫu dữ liệu...")
        # hàm dưới là xáo trộn ngẫu nhiên và chia tỉ lệ 80/20, random_state=42 giúp mỗi lần chạy nó chia đều giống nhau để so sánh kết quả
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_hog = self._extract_hog_features(X_train)
        X_test_hog = self._extract_hog_features(X_test)
        
        # hàm fit là hàm sẽ tính công thức Z và softmax
        self.model.fit(X_train_hog, y_train)
        # học xong cho cờ true
        self.is_trained = True
        
        # kiểm tra với 20% cái vừa rồi
        preds = self.model.predict(X_test_hog) # cất vô biến preds
        acc = accuracy_score(y_test, preds) # so sánh với đáp án thực tế
        print(f"Training hoàn tất. Độ chính xác trên tập test: {acc * 100:.2f}%")
        # lấy độ chính xác ra để lưu
        return (acc * 100)

    # hàm dự đoán
    def predict(self, X_input):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        # đưa ma trận ảnh đã flatten của người dùng vào mô hình
        # mô hình sẽ dùng bộ trọng số để tính xác suất và gán nhãn
        X_hog = self._extract_hog_features(X_input)
        return self.model.predict(X_hog)

    def predict_proba(self, X_input):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        X_hog = self._extract_hog_features(X_input)
        return self.model.predict_proba(X_hog)

    # hàm lưu và tải mô hình
    def save(self, filepath):
        payload = {
            "version": "hog_v1",
            "model": self.model
        }
        joblib.dump(payload, filepath)
        print(f"Đã lưu model tại: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            loaded_obj = joblib.load(filepath)

            if isinstance(loaded_obj, dict) and loaded_obj.get("version") == "hog_v1":
                self.model = loaded_obj["model"]
                self.is_trained = True
                print("Đã load model HOG thành công.")
                return True

            print("Model cũ không đúng định dạng HOG, sẽ train lại model mới.")
        return False