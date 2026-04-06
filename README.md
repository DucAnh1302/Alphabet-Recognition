# Alphabet Recognition Using Logistic Regression

Tài liệu này mô tả lại toàn bộ bài toán nhận dạng chữ cái viết tay (A-Z) theo đúng code hiện tại của nhóm, theo mô hình 3 lớp, có công thức và ví dụ thế số.

## 0. Mục tiêu bài toán

- Input: ảnh chữ cái viết tay trong thư mục Input.
- Output: nhãn chữ cái dự đoán (A-Z), ghi ra Output/result.txt.
- Mô hình học máy: Multiclass Logistic Regression (Softmax), đặc trưng HOG.
- Kết quả thực nghiệm gần nhất: accuracy test khoảng 95.52%.

## 1. Ý tưởng cốt lõi: từ Linear đến Logistic

### 1.1. Vì sao phải đi từ Linear

Mỗi lớp (A, B, ..., Z) được gán một hàm điểm tuyến tính:

$$
z_k = \mathbf{w}_k^T \mathbf{x} + b_k
$$

- $\mathbf{x}$: vector đặc trưng của ảnh (ở dự án này là HOG).
- $\mathbf{w}_k, b_k$: tham số của lớp $k$.
- $z_k$: điểm thô (linear score) của lớp $k$.

Linear score giúp xếp hạng lớp nào "hợp" với ảnh hơn, nhưng chưa phải xác suất.

### 1.2. Logistic (Softmax) để ra xác suất

Từ các điểm $z_k$, dùng Softmax để chuyển thành xác suất:

$$
P(y=k\mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad K=26
$$

Ưu điểm:

- Tổng xác suất 26 lớp bằng 1.
- Có thể giải thích theo xác suất (ví dụ lớp A 92.1%).

### 1.3. Hàm mất mát khi huấn luyện

Với một mẫu có nhãn đúng $y$, dùng cross-entropy:

$$
\mathcal{L} = -\log P(y\mid \mathbf{x})
$$

Mô hình học bằng cách tối ưu để xác suất lớp đúng càng cao càng tốt.

## 2. Mô hình 3 lớp trong dự án

## 2.1. Tổng quan 3 lớp

1. Presentation Layer (điều phối vào/ra): file main.py
2. Business/Model Layer (logic ML): file src/model_logic.py
3. Data Layer (đọc dữ liệu, tiền xử lý): file src/data_loader.py và src/config.py

Luồng tổng quát:

1. main.py kiểm tra model cũ.
2. Nếu chưa có model đúng định dạng thì gọi DataLoader đọc dữ liệu train.
3. Model layer train Logistic Regression trên HOG.
4. Đọc ảnh input mới, preprocess, predict.
5. Ghi kết quả ra Output/result.txt.

## 2.2. Vai trò từng lớp

### Presentation Layer (main.py)

- Quản lý tiến trình chạy chương trình.
- Quyết định train mới hay load model cũ.
- Gọi predict cho ảnh input.
- Ghi log thời gian và kết quả dự đoán.

### Business/Model Layer (src/model_logic.py)

- Định nghĩa mô hình Logistic Regression đa lớp.
- Trích xuất HOG từ ảnh (chuyển từ vector 784 sang vector HOG).
- Train/test split và đánh giá accuracy.
- Save/load model có version (hog_v1) để tránh dùng model cũ sai định dạng.

### Data Layer (src/data_loader.py, src/config.py)

- Đọc ảnh từ Data và Input.
- Tiền xử lý ảnh về chuẩn 28x28 và chuẩn hóa [0,1].
- Với ảnh Input: xử lý nhiễu, threshold, crop contour, center, dilation.
- Cấu hình đường dẫn và kích thước ảnh trong config.py.

## 3. Quy trình chi tiết: input ảnh -> xử lý -> HOG -> linear -> logistic -> output

## 3.1. Giai đoạn A: Tiền xử lý ảnh

### A1. Chuẩn hóa kích thước

Ảnh sau xử lý về kích thước cố định:

$$
28 \times 28
$$

Giúp đầu vào nhất quán cho model.

### A2. Chuẩn hóa giá trị pixel

Từ ảnh xám 8-bit (0..255), đưa về [0,1]:

$$
x_i = \frac{p_i}{255}
$$

Ví dụ thế số:

- Pixel đen gần như hoàn toàn: $p_i=0 \Rightarrow x_i=0$.
- Pixel trắng hoàn toàn: $p_i=255 \Rightarrow x_i=1$.
- Pixel xám: $p_i=128 \Rightarrow x_i=\frac{128}{255}\approx 0.502$.

### A3. Xử lý riêng cho ảnh Input (ảnh người dùng)

Trong data_loader.py, với is_input=True có thêm:

1. Kiểm tra nền sáng/tối và đảo màu khi cần.
2. Gaussian blur để giảm nhiễu.
3. Adaptive threshold để nhị phân hóa nét chữ.
4. Tìm contour lớn nhất, cắt ROI chữ.
5. Căn giữa lên canvas vuông.
6. Resize về 28x28 và làm dày nét (dilate nhẹ).

Mục tiêu: ảnh input thực tế gần với phân phối ảnh train hơn.

## 3.2. Giai đoạn B: Trích xuất đặc trưng HOG

Model không học trực tiếp trên pixel thô mà học trên đặc trưng HOG.

Thông số HOG trong code:

- winSize: (28,28)
- blockSize: (14,14)
- blockStride: (7,7)
- cellSize: (7,7)
- nbins: 9

### B1. Ý tưởng HOG

HOG đo hướng gradient (biên cạnh) của nét chữ theo từng vùng nhỏ.

Các bước khái niệm:

1. Tính gradient theo trục x, y:

$$
g_x = I(x+1,y)-I(x-1,y), \quad g_y = I(x,y+1)-I(x,y-1)
$$

2. Độ lớn và hướng gradient:

$$
m = \sqrt{g_x^2 + g_y^2}, \quad \theta = \text{atan2}(g_y,g_x)
$$

3. Bỏ phiếu vào histogram hướng (9 bins) theo từng cell.
4. Chuẩn hóa theo block để giảm ảnh hưởng sáng/tối.

### B2. Ví dụ thế số gradient

Giả sử tại 1 điểm ảnh:

- $g_x = 30$
- $g_y = 40$

Khi đó:

$$
m = \sqrt{30^2 + 40^2} = 50
$$

$$
	heta = \text{atan2}(40,30) \approx 53.13^\circ
$$

Điểm ảnh này sẽ đóng góp cường độ 50 vào bin hướng gần $53^\circ$.

## 3.3. Giai đoạn C: Linear score cho 26 lớp

Sau HOG, mỗi ảnh thành vector đặc trưng $\mathbf{x}_{hog}$ (chiều khoảng vài trăm phần tử, theo OpenCV HOG hiện tại là 324).

Với mỗi lớp $k \in \{A,...,Z\}$:

$$
z_k = \mathbf{w}_k^T \mathbf{x}_{hog} + b_k
$$

Ví dụ rút gọn (chỉ minh họa 3 đặc trưng đầu):

- $\mathbf{x}=[0.5, 1.2, -0.7]$
- Với lớp A: $\mathbf{w}_A=[0.8, -0.4, 0.3],\ b_A=0.1$

$$
z_A = 0.8\cdot0.5 + (-0.4)\cdot1.2 + 0.3\cdot(-0.7) + 0.1
$$

$$
z_A = 0.4 - 0.48 - 0.21 + 0.1 = -0.19
$$

Làm tương tự cho B..Z để có 26 điểm thô.

## 3.4. Giai đoạn D: Logistic (Softmax) ra xác suất

Giả sử top-3 điểm thô:

- $z_A = 2.0$
- $z_B = 1.0$
- $z_C = 0.1$

(Các lớp khác nhỏ hơn nhiều, minh họa bỏ qua)

Tính mũ:

$$
e^{2.0}=7.389,\quad e^{1.0}=2.718,\quad e^{0.1}=1.105
$$

Tổng:

$$
S=7.389+2.718+1.105=11.212
$$

Xác suất:

$$
P(A\mid x)=\frac{7.389}{11.212}=0.659\ (65.9\%)
$$

$$
P(B\mid x)=\frac{2.718}{11.212}=0.242\ (24.2\%)
$$

$$
P(C\mid x)=\frac{1.105}{11.212}=0.099\ (9.9\%)
$$

Dự đoán cuối cùng là lớp có xác suất lớn nhất (ở đây là A).

## 3.5. Giai đoạn E: Output

- In dự đoán ra console.
- Ghi vào Output/result.txt gồm:
	- thời gian,
	- accuracy train (nếu vừa train mới),
	- danh sách tên file và nhãn dự đoán.

## 4. Phân tích các file code trong dự án

## 4.1. Nhóm file runtime chính

1. main.py
- Entry point của hệ thống.
- Điều phối train/load/predict và ghi output.

2. src/config.py
- Khai báo hằng số cấu hình:
	- IMG_SIZE
	- DATA_PATH, INPUT_PATH, OUTPUT_PATH, MODEL_PATH

3. src/data_loader.py
- preprocess_image(image_path, is_input=False): xử lý ảnh.
- load_training_data(root_folder, max_per_class): đọc dữ liệu train A-Z.
- load_input_folder(input_folder): đọc ảnh cần dự đoán.

4. src/model_logic.py
- Lớp AlphabetRecognizer.
- Tạo HOGDescriptor.
- _extract_hog_features / transform_features.
- train / predict / predict_proba.
- save / load (format version hog_v1).

## 4.2. Nhóm file test minh họa

1. test/MainTest_0_Preprocess.py
- Kiểm tra ảnh sau preprocess và histogram pixel.

2. test/MainTest_1_LinearScore.py
- Minh họa điểm tuyến tính Z của các lớp cao nhất.

3. test/MainTest_2_Probability.py
- Minh họa xác suất Softmax top-k.

4. test/MainTest_3_WeightsViz.py
- Trực quan trọng số học được của lớp dự đoán.

## 4.3. Nhóm file chuẩn bị dữ liệu

1. ToolKaggleToImages.py
- Chuyển EMNIST CSV thành ảnh theo thư mục lớp.

2. ToolCreateFolders.py
- Tạo cấu trúc thư mục dữ liệu.

## 5. Ví dụ luồng thực tế end-to-end

Giả sử có ảnh Input/Hinh1.jpg:

1. main.py gọi load_input_folder.
2. data_loader.py preprocess ảnh về vector 784 chuẩn hóa.
3. model_logic.py đổi 784 -> HOG (324).
4. Tính 26 linear score $z_k$.
5. Softmax ra 26 xác suất.
6. Lấy lớp max, ví dụ J.
7. Ghi dòng: Hinh1.jpg | J vào result.txt.

## 6. Vì sao HOG + Logistic hiệu quả hơn pixel thô

- Pixel thô nhạy với dịch chuyển, độ dày nét, nhiễu.
- HOG tập trung vào hướng nét (edge orientation), phù hợp nhận dạng ký tự.
- Logistic Regression học ranh giới tuyến tính trên không gian đặc trưng giàu thông tin hơn.

Do đó accuracy tăng mạnh so với dùng trực tiếp 784 pixel.

## 7. Gợi ý mở rộng

1. Thử tăng dữ liệu mỗi lớp (max_per_class) để đánh đổi thời gian/độ chính xác.
2. Thêm confusion matrix để phân tích cặp chữ dễ nhầm (ví dụ I-L, O-Q).
3. Bổ sung đánh giá precision/recall/F1 theo từng lớp.
4. So sánh thêm SVM hoặc MLP trên cùng đặc trưng HOG.

## 8. Công thức tóm tắt nhanh theo từng giai đoạn

1. Chuẩn hóa pixel:

$$
x_i = \frac{p_i}{255}
$$

2. Gradient HOG:

$$
g_x = I(x+1,y)-I(x-1,y),\quad g_y = I(x,y+1)-I(x,y-1)
$$

$$
m = \sqrt{g_x^2+g_y^2},\quad \theta = \text{atan2}(g_y,g_x)
$$

3. Linear score:

$$
z_k = \mathbf{w}_k^T\mathbf{x}_{hog}+b_k
$$

4. Softmax logistic:

$$
P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{26}e^{z_j}}
$$

5. Loss (cross-entropy):

$$
\mathcal{L}=-\log P(y_{true}\mid x)
$$

6. Accuracy:

$$
	ext{Accuracy}=\frac{\#\text{dự đoán đúng}}{\#\text{tổng mẫu}}\times 100\%
$$

---

Nếu cần, có thể tách README này thành 2 tài liệu:

1. README ngắn cho chạy project.
2. REPORT.md dài cho phần lý thuyết + công thức + ví dụ tính tay.
