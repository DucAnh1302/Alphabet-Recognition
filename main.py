# main.py
import tkinter as tk
from src.gui import DrawingApp

def main():
    print("Đang khởi động Giao diện Whiteboard...")
    root = tk.Tk()
    app = DrawingApp(root)

    # Luon thu hoi tai nguyen khi dong cua so hoac dung chuong trinh.
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Đã nhận tín hiệu dừng chương trình.")
    finally:
        app.shutdown_resources()
        try:
            if root.winfo_exists():
                root.destroy()
        except tk.TclError:
            pass

if __name__ == "__main__":
    main()