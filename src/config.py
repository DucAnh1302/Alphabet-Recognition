# src/config.py
import os
import sys

# Kích thước ảnh chuẩn hóa (Logistic Regression cần input cố định)
IMG_SIZE = (28, 28)


def _get_base_dir():
	"""Thư mục chứa app khi đóng gói hoặc thư mục dự án khi chạy source."""
	if getattr(sys, "frozen", False):
		return os.path.dirname(sys.executable)
	return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_user_data_dir():
	"""Thư mục luôn có quyền ghi để lưu file tạm và ảnh debug."""
	local_appdata = os.environ.get("LOCALAPPDATA")
	if local_appdata:
		return os.path.join(local_appdata, "AlphabetRecognitionUsingLogisticRegression")
	return os.path.join(os.path.expanduser("~"), ".alphabet_recognition")

# Đường dẫn runtime
BASE_DIR = _get_base_dir()
USER_DATA_DIR = _get_user_data_dir()
DATA_PATH = os.path.join(BASE_DIR, 'Data')
INPUT_PATH = os.path.join(USER_DATA_DIR, 'Input')
OUTPUT_PATH = os.path.join(USER_DATA_DIR, 'Output')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl') # Để lưu model đã train


def ensure_runtime_dirs():
	os.makedirs(INPUT_PATH, exist_ok=True)
	os.makedirs(OUTPUT_PATH, exist_ok=True)