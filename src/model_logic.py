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
            max_iter=3000, # số vòng lặp tối đa để thuật toán tìm ra trọng số W, tại nếu không có cái này là nó lặp đến vô hạn
            #solver='saga', # đại khái là thư viện này có nhiều loại (libinear, lbfgs,..) nhưng saga là thuật toán ok nhất cho dữ liệu lớn và phân loại 
            # Đổi 'saga' sang 'lbfgs' (hội tụ nhanh hơn với dữ liệu < 50k)
            solver = 'lbfgs',
            n_jobs=-1, # ép máy tính sử dụng nhân của CPU để tính song song để tăng tốc độ
            #tol=0.1, # nếu chênh lệch lỗi < 0.1 sẽ dừng sớm tránh mất thời gian
            #C=0.1, # giúp mô hình chống học vẹt á
            C = 20.0, # nâng lên 1.0 để nới lỏng hơn và không cần tol=0.1
            class_weight='balanced' # ví dụ số lượng ảnh của mình là 5000 ảnh chữ a và chỉ có 1000 ảnh chữ z thì hàm này cần bằng lại
        )
        self.is_trained = False # cờ để kiểm tra mô hình train chưa thôi

    # nhập số lượng ảnh x và đáp án y để học
    def train(self, X, y):
        print(f"Bắt đầu training với {len(X)} mẫu dữ liệu...")
        # hàm dưới là xáo trộn ngẫu nhiên và chia tỉ lệ 80/20, random_state=42 giúp mỗi lần chạy nó chia đều giống nhau để so sánh kết quả
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # hàm fit là hàm sẽ tính công thức Z và softmax
        self.model.fit(X_train, y_train)
        # học xong cho cờ true
        self.is_trained = True
        
        # kiểm tra với 20% cái vừa rồi
        preds = self.model.predict(X_test) # cất vô biến preds
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
        return self.model.predict(X_input)

    # hàm lưu và tải mô hình
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