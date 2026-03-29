# src/model_logic.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class AlphabetRecognizer:
    def __init__(self):
        # THÊM C=0.1 VÀ class_weight='balanced' ĐỂ CHỐNG ĐOÁN MÒ
        self.model = LogisticRegression(
            max_iter=1000, 
            solver='saga', 
            n_jobs=-1,
            tol=0.1,
            C=0.1,
            class_weight='balanced'
        )
        self.is_trained = False

    def train(self, X, y):
        print(f"Bắt đầu training với {len(X)} mẫu dữ liệu...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Training hoàn tất. Độ chính xác trên tập test: {acc * 100:.2f}%")
        # lấy độ chính xác ra để lưu
        return (acc * 100)

    def predict(self, X_input):
        if not self.is_trained:
            raise Exception("Model chưa được train!")
        return self.model.predict(X_input)

    def save(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"Đã lưu model tại: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            print("Đã load model cũ thành công.")
            return True
        return False