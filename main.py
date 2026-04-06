# main.py
import tkinter as tk
from src.gui import DrawingApp

def main():
    print("Đang khởi động Giao diện Whiteboard...")
    root = tk.Tk()
    app = DrawingApp(root)
    
    # Lệnh này giữ cho cửa sổ ứng dụng mở liên tục
    root.mainloop() 

if __name__ == "__main__":
    main()