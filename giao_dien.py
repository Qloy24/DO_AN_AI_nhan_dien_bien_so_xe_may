import tkinter as tk
from tkinter import Label, Button, Frame, Scrollbar, ttk
from PIL import Image, ImageTk
import cv2

# =====================================================================
# LỚP PlateUI — Giao diện chính của hệ thống nhận diện biển số
# =====================================================================
# Lớp này chịu trách nhiệm:
#   - Tạo cửa sổ giao diện (UI) bằng Tkinter.
#   - Hiển thị luồng video từ camera.
#   - Hiển thị biển số được nhận diện.
#   - Ghi lại lịch sử các biển số đã phát hiện.
# =====================================================================
class PlateUI:
    def __init__(self, detector):
        """
        Hàm khởi tạo giao diện.
        :param detector: Đối tượng 'detector' có nhiệm vụ xử lý camera và nhận diện biển số.
        """
        self.detector = detector                      # Đối tượng xử lý nhận diện biển số
        self.window = tk.Tk()                         # Khởi tạo cửa sổ chính
        self.window.title("Nhận diện biển số xe")     # Tiêu đề cửa sổ
        self.window.geometry("1000x850")              # Kích thước mặc định cửa sổ

        # Gọi hàm xây dựng giao diện
        self._build_ui()

    # -----------------------------------------------------------------
    # XÂY DỰNG GIAO DIỆN (UI)
    # -----------------------------------------------------------------
    def _build_ui(self):
        """Tạo các khung giao diện, bảng lịch sử, nhãn và nút điều khiển."""
        # ======= KHUNG HIỂN THỊ CAMERA =======
        self.camera_frame = Frame(self.window, width=640, height=480, bg="black")
        self.camera_frame.pack(pady=10)
        self.camera_frame.pack_propagate(False)  # Không cho khung tự điều chỉnh kích thước

        # Nhãn (Label) để hiển thị ảnh từ camera
        self.camera_label = Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill="both", expand=True)

        # ======= NHÃN HIỂN THỊ KẾT QUẢ BIỂN SỐ =======
        self.plate_label = Label(
            self.window,
            text="Chưa nhận diện biển số nào",
            fg="blue",
            font=("Arial", 12, "bold")
        )
        self.plate_label.pack(pady=5)

        # ======= BẢNG LỊCH SỬ NHẬN DIỆN =======
        Label(self.window, text="Lịch sử nhận diện:", font=("Arial", 11, "bold")).pack(pady=(5, 0))
        frame_history = Frame(self.window, bg="white", bd=1, relief="solid")
        frame_history.pack(pady=5)

        # Thanh cuộn dọc (scrollbar)
        scrollbar_y = Scrollbar(frame_history)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Bảng Treeview để hiển thị dữ liệu lịch sử
        self.history_table = ttk.Treeview(
            frame_history,
            columns=("time", "plate"),  # Hai cột: thời gian và biển số
            show="headings",
            yscrollcommand=scrollbar_y.set,
            height=8
        )
        self.history_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self.history_table.yview)

        # Đặt tiêu đề cột
        self.history_table.heading("time", text="Thời gian")
        self.history_table.heading("plate", text="Biển số")

        # Đặt độ rộng cột và căn giữa
        self.history_table.column("time", width=220, anchor="center")
        self.history_table.column("plate", width=350, anchor="center")

        # Tùy chỉnh giao diện bảng
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", font=("Arial", 10), rowheight=25)

        # ======= CÁC NÚT CHỨC NĂNG =======
        button_frame = Frame(self.window)
        button_frame.pack(pady=10)

        # Nút bắt đầu camera
        self.btn_start = Button(button_frame, text="Bắt đầu", width=15, command=self.start_camera)
        # Nút dừng camera
        self.btn_stop = Button(button_frame, text="Dừng", width=15, command=self.stop_camera)
        # Nút thoát chương trình
        self.btn_exit = Button(button_frame, text="Thoát", width=15, command=self.window.destroy)

        # Bố trí nút theo hàng
        self.btn_start.grid(row=0, column=0, padx=10)
        self.btn_stop.grid(row=0, column=1, padx=10)
        self.btn_exit.grid(row=0, column=2, padx=10)

    # -----------------------------------------------------------------
    # CHỨC NĂNG CAMERA
    # -----------------------------------------------------------------
    def start_camera(self):
        """Bắt đầu đọc và hiển thị luồng video từ camera."""
        self.detector.running = True
        self.update_frame()  # Cập nhật khung hình liên tục

    def stop_camera(self):
        """Dừng luồng camera."""
        self.detector.running = False

    # -----------------------------------------------------------------
    # CẬP NHẬT KHUNG HÌNH LIÊN TỤC
    # -----------------------------------------------------------------
    def update_frame(self):
        """
        Hàm này đọc từng khung hình từ camera, xử lý nhận diện biển số,
        hiển thị kết quả và cập nhật giao diện liên tục.
        """
        if not self.detector.running:
            return  # Nếu camera đã dừng thì thoát

        # Đọc khung hình từ camera
        ret, frame = self.detector.cap.read()
        if not ret:
            return  # Không đọc được khung hình thì dừng

        # Gọi hàm nhận diện biển số
        current_plate = self.detector.detect_plate(frame)

        # Gọi hàm ổn định kết quả (lọc nhiễu, đảm bảo độ chính xác)
        result = self.detector.stabilize_plate(current_plate)

        # Nếu có kết quả nhận diện hợp lệ
        if result:
            # Hiển thị biển số và thông tin chủ xe lên giao diện
            self.plate_label.config(
                text=f"Biển số: {result['plate']} (Chủ xe: {result['owner']} - {result['location']})"
            )

            # Thêm dữ liệu vào bảng lịch sử
            self.history_table.insert(
                "",
                "end",
                values=(result["time"], f"{result['plate']} ({result['owner']} - {result['location']})")
            )

            # Tự động cuộn xuống dòng cuối
            self.history_table.yview_moveto(1.0)

        # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB (hiển thị Tkinter)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize ảnh về kích thước khung camera
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Chuyển ảnh sang định dạng ImageTk để hiển thị
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        # Gọi lại hàm update_frame sau 10ms để tạo hiệu ứng video
        self.camera_label.after(10, self.update_frame)

    # -----------------------------------------------------------------
    # CHẠY CHƯƠNG TRÌNH
    # -----------------------------------------------------------------
    def run(self):
        """Khởi chạy vòng lặp chính của giao diện và giải phóng tài nguyên khi thoát."""
        self.window.mainloop()
        self.detector.release()  # Giải phóng camera sau khi đóng cửa sổ