import tkinter as tk
from tkinter import Label, Button, Frame, Scrollbar, ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

class PlateUI:
    def __init__(self, detector):
        self.detector = detector
        self.window = tk.Tk()
        self.window.title("Nhận diện biển số xe")
        self.window.geometry("1300x850")

        self._build_ui()

    # ---------------------------------------------------------------
    # XÂY DỰNG GIAO DIỆN
    # ---------------------------------------------------------------
    def _build_ui(self):
        top_frame = Frame(self.window)
        top_frame.pack(pady=10)

        # --- Camera Frame ---
        self.camera_frame = Frame(top_frame, width=640, height=480, bg="black")
        self.camera_frame.pack(side=tk.LEFT, padx=10)
        self.camera_frame.pack_propagate(False)
        self.camera_label = Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill="both", expand=True)

        # --- Upload Frame ---
        self.upload_frame = Frame(top_frame, width=640, height=480, bg="gray")
        self.upload_frame.pack(side=tk.LEFT, padx=10)
        self.upload_frame.pack_propagate(False)
        self.upload_label = Label(self.upload_frame, bg="gray", text="Chưa có ảnh", fg="white")
        self.upload_label.pack(fill="both", expand=True)

        # --- Kết quả ---
        self.plate_label = Label(
            self.window,
            text="Chưa nhận diện biển số nào",
            fg="blue",
            font=("Arial", 12, "bold")
        )
        self.plate_label.pack(pady=5)

        # --- Lịch sử ---
        Label(self.window, text="Lịch sử nhận diện:", font=("Arial", 11, "bold")).pack(pady=(5, 0))
        frame_history = Frame(self.window, bg="white", bd=1, relief="solid")
        frame_history.pack(pady=5)

        scrollbar_y = Scrollbar(frame_history)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_table = ttk.Treeview(
            frame_history,
            columns=("time", "plate"),
            show="headings",
            yscrollcommand=scrollbar_y.set,
            height=8
        )
        self.history_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self.history_table.yview)

        self.history_table.heading("time", text="Thời gian")
        self.history_table.heading("plate", text="Biển số")
        self.history_table.column("time", width=220, anchor="center")
        self.history_table.column("plate", width=350, anchor="center")

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", font=("Arial", 10), rowheight=25)

        # --- Nút chức năng ---
        button_frame = Frame(self.window)
        button_frame.pack(pady=10)

        self.btn_start = Button(button_frame, text="Bắt đầu", width=15, command=self.start_camera)
        self.btn_stop = Button(button_frame, text="Dừng", width=15, command=self.stop_camera)
        self.btn_upload = Button(button_frame, text="Tải ảnh", width=15, command=self.load_image)
        self.btn_exit = Button(button_frame, text="Thoát", width=15, command=self.window.destroy)

        self.btn_start.grid(row=0, column=0, padx=10)
        self.btn_stop.grid(row=0, column=1, padx=10)
        self.btn_upload.grid(row=0, column=2, padx=10)
        self.btn_exit.grid(row=0, column=3, padx=10)

    # ---------------------------------------------------------------
    # CAMERA
    # ---------------------------------------------------------------
    def start_camera(self):
        self.detector.running = True
        self.update_frame()

    def stop_camera(self):
        self.detector.running = False

    # ---------------------------------------------------------------
    # TẢI ẢNH VÀ NHẬN DIỆN
    # ---------------------------------------------------------------
    def load_image(self):
        """Chọn file ảnh, hiển thị và nhận diện biển số."""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return

        # --- Nhận diện ---
        result = self.detector.recognize_from_image(file_path)
        if not result:
            messagebox.showwarning("Không phát hiện", "Không tìm thấy biển số nào trong ảnh.")
            self.upload_label.config(image="", text="Không nhận diện được", fg="red")
            return

        # --- Hiển thị ảnh có vẽ khung ---
        frame_rgb = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)

        self.upload_label.imgtk = imgtk
        self.upload_label.configure(image=imgtk, text="")

        # --- Hiển thị kết quả ---
        plate_text = f"Biển số: {result['plate']} (Chủ xe: {result['owner']} - {result['location']})"
        self.plate_label.config(text=plate_text)

        # --- Lưu vào lịch sử ---
        self.history_table.insert(
            "",
            "end",
            values=(result["time"], f"{result['plate']} ({result['owner']} - {result['location']})")
        )
        self.history_table.yview_moveto(1.0)

    # ---------------------------------------------------------------
    # CẬP NHẬT CAMERA FRAME
    # ---------------------------------------------------------------
    def update_frame(self):
        if not self.detector.running:
            return
        ret, frame = self.detector.cap.read()
        if not ret:
            return

        current_plate = self.detector.detect_plate(frame)
        result = self.detector.stabilize_plate(current_plate)

        if result:
            self.plate_label.config(
                text=f"Biển số: {result['plate']} (Chủ xe: {result['owner']} - {result['location']})"
            )
            self.history_table.insert(
                "",
                "end",
                values=(result["time"], f"{result['plate']} ({result['owner']} - {result['location']})")
            )
            self.history_table.yview_moveto(1.0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        self.camera_label.after(10, self.update_frame)

    # ---------------------------------------------------------------
    # CHẠY ỨNG DỤNG
    # ---------------------------------------------------------------
    def run(self):
        self.window.mainloop()
        self.detector.release()
