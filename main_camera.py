import cv2
from ultralytics import YOLO
import easyocr
from bien_so_map_dau import BIEN_SO_MAP_DAU
from bien_so_map import BIEN_SO_MAP
from chu_xe import CHU_XE
import tkinter as tk
from tkinter import Label, Button, Frame, Scrollbar, ttk
from PIL import Image, ImageTk
import datetime
import random

# =====================================================================
# 1 CẤU HÌNH MÔ HÌNH & OCR
# Khởi tạo mô hình YOLO và EasyOCR để phát hiện biển số và nhận diện chữ.
# =====================================================================
MODEL_PATH = "D:/Do_An_AI/runs/detect/train_bien_so_100epoch/weights/best.pt" #thay đổi đường dẫn tới mô hình của bạn
model = YOLO(MODEL_PATH)  # Tải mô hình YOLO đã huấn luyện trước để phát hiện biển số
reader = easyocr.Reader(['vi'], gpu=True)  # Khởi tạo EasyOCR cho nhận diện chữ tiếng Việt

# =====================================================================
# 2 KHỞI TẠO GIAO DIỆN NGƯỜI DÙNG
# Thiết lập giao diện đồ họa dựa trên Tkinter để hiển thị luồng camera,
# kết quả biển số và lịch sử nhận diện.
# =====================================================================
window = tk.Tk()
window.title("Nhận diện biển số xe")  # Đặt tiêu đề cửa sổ
window.geometry("1000x850")  # Đặt kích thước cửa sổ là 1000x850 pixel

# --- Khung hiển thị camera ---
camera_frame = Frame(window, width=640, height=480, bg="black")
camera_frame.pack(pady=10)
camera_frame.pack_propagate(False)  # Ngăn khung thay đổi kích thước theo nội dung

camera_label = Label(camera_frame, bg="black")  # Nhãn để hiển thị luồng camera trực tiếp
camera_label.pack(fill="both", expand=True)

# --- Hiển thị kết quả biển số ---
plate_label = Label(
    window,
    text="Chưa nhận diện biển số nào",  # Văn bản mặc định ban đầu
    fg="blue",
    font=("Arial", 12, "bold")
)
plate_label.pack(pady=5)

# --- Phần lịch sử nhận diện ---
Label(window, text="Lịch sử nhận diện:", font=("Arial", 11, "bold")).pack(pady=(5, 0))

frame_history = Frame(window, bg="white", bd=1, relief="solid")  # Khung cho bảng lịch sử
frame_history.pack(pady=5)

scrollbar_y = Scrollbar(frame_history)  # Thanh cuộn dọc cho bảng lịch sử
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

history_table = ttk.Treeview(
    frame_history,
    columns=("time", "plate"),
    show="headings",
    yscrollcommand=scrollbar_y.set,
    height=8
)  # Bảng để hiển thị lịch sử nhận diện
history_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_y.config(command=history_table.yview)

# --- Cấu hình bảng lịch sử ---
history_table.heading("time", text="Thời gian")  # Tiêu đề cột thời gian
history_table.heading("plate", text="Biển số")   # Tiêu đề cột biển số

history_table.column("time", width=220, anchor="center")  # Đặt chiều rộng và căn giữa cột
history_table.column("plate", width=350, anchor="center")

# --- Cấu hình giao diện bảng ---
style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 10, "bold"))  # Kiểu chữ cho tiêu đề bảng
style.configure("Treeview", font=("Arial", 10), rowheight=25)    # Kiểu chữ và chiều cao hàng
style.map("Treeview", background=[("selected", "#cce5ff")])      # Màu nền khi chọn hàng

# --- Nút điều khiển ---
button_frame = Frame(window)  # Khung cho các nút điều khiển
button_frame.pack(pady=10)

btn_start = Button(button_frame, text="Bắt đầu", width=15)  # Nút bắt đầu camera
btn_stop = Button(button_frame, text="Dừng", width=15)       # Nút dừng camera
btn_exit = Button(button_frame, text="Thoát", width=15, command=window.destroy)  # Nút thoát ứng dụng

btn_start.grid(row=0, column=0, padx=10)
btn_stop.grid(row=0, column=1, padx=10)
btn_exit.grid(row=0, column=2, padx=10)

# =====================================================================
# 3 CAMERA & BIẾN TOÀN CỤC
# Khởi tạo camera và các biến toàn cục để kiểm soát quá trình nhận diện
# và lưu trữ kết quả.
# =====================================================================
cap = cv2.VideoCapture(0)  # Khởi tạo webcam (index 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Đặt độ phân giải chiều rộng camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Đặt độ phân giải chiều cao camera

running = False  # Cờ để kiểm soát vòng lặp camera
frame_count = 0  # Bộ đếm khung hình được xử lý
last_results = []  # Bộ nhớ đệm cho kết quả phát hiện YOLO
plate_buffer = []  # Bộ đệm để ổn định kết quả nhận diện biển số
last_confirmed_plate = ""  # Biển số được xác nhận cuối cùng để tránh trùng lặp
plate_owner_map = {}  # Lưu ánh xạ biển số -> tên chủ xe

# =====================================================================
# 4 CÁC HÀM XỬ LÝ CHÍNH
# Chứa các hàm để khởi động/dừng camera, phát hiện biển số, ổn định kết quả
# và cập nhật giao diện.
# =====================================================================
def start_camera():
    """Bắt đầu luồng camera và quá trình nhận diện biển số."""
    global running
    running = True
    update_frame()  # Bắt đầu cập nhật khung hình

def stop_camera():
    """Dừng luồng camera và quá trình nhận diện."""
    global running
    running = False

def detect_plate(frame):
    """
    Phát hiện biển số trong khung hình sử dụng YOLO và trích xuất văn bản
    bằng EasyOCR.

    Args:
        frame: Khung hình đầu vào từ camera (định dạng BGR)

    Returns:
        str: Văn bản biển số được phát hiện (rỗng nếu không có)
    """
    global last_results, frame_count

    frame_count += 1
    # Xử lý mỗi khung hình thứ 5 để giảm tải tính toán
    if frame_count % 15 == 0:
        results = model(frame)  # Chạy mô hình YOLO để phát hiện
        last_results = results
    else:
        results = last_results  # Sử dụng kết quả đã lưu để tăng hiệu quả

    current_plate = ""

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ khung bao
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung bao quanh biển số

            bien_so_crop = frame[y1:y2, x1:x2]  # Cắt vùng biển số
            if bien_so_crop.size == 0:
                continue  # Bỏ qua nếu vùng cắt rỗng

            result = reader.readtext(bien_so_crop)  # Thực hiện OCR trên vùng cắt
            text = " ".join([res[1] for res in result]) if result else ""  # Kết hợp kết quả OCR

            if text:
                # Trích xuất mã tỉnh (phần đầu trước '-' hoặc 2 ký tự đầu)
                ma_tinh = text.split("-")[0] if "-" in text else text[:2]
                dia_phuong = BIEN_SO_MAP.get(ma_tinh, "Không rõ địa phương")  # Ánh xạ mã tỉnh

                current_plate = text
                # Hiển thị biển số và địa phương trên khung hình
                cv2.putText(
                    frame,
                    f"{text} ({dia_phuong})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    return current_plate

def stabilize_plate(current_plate):
    """
    Ổn định kết quả nhận diện biển số bằng cách yêu cầu kết quả nhất quán
    để xác nhận và tránh trùng lặp, đồng thời gán tên chủ xe ngẫu nhiên
    (nhưng cố định cho mỗi biển).
    """
    global plate_buffer, last_confirmed_plate, plate_owner_map

    if not current_plate:
        return

    # Thêm vào bộ đệm (lưu 5 kết quả gần nhất)
    plate_buffer.append(current_plate)
    if len(plate_buffer) > 5:
        plate_buffer.pop(0)

    # Xác nhận khi có ít nhất 3 lần liên tiếp cùng 1 biển số
    if plate_buffer.count(current_plate) >= 3 and current_plate != last_confirmed_plate:
        ma_tinh = current_plate.split("-")[0] if "-" in current_plate else current_plate[:2]
        dia_phuong = BIEN_SO_MAP_DAU.get(ma_tinh, "Không rõ địa phương")

        # --- Gán tên chủ xe (ngẫu nhiên 1 lần duy nhất cho biển đó) ---
        if current_plate not in plate_owner_map:
            plate_owner_map[current_plate] = random.choice(CHU_XE)
        ten_chu_xe = plate_owner_map[current_plate]

        # --- Cập nhật giao diện ---
        plate_label.config(
            text=f"Biển số: {current_plate} (Chủ xe: {ten_chu_xe} - {dia_phuong})",
            fg="blue"
        )

        # --- Ghi vào bảng lịch sử ---
        now = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        history_table.insert("", "end", values=(now, f"{current_plate} ({ten_chu_xe} - {dia_phuong})"))
        history_table.yview_moveto(1.0)

        last_confirmed_plate = current_plate

def update_frame():
    """
    Cập nhật luồng camera trong giao diện và xử lý từng khung hình để
    nhận diện biển số.
    """
    if not running:
        return

    ret, frame = cap.read()  # Đọc khung hình từ camera
    if not ret:
        return  # Thoát nếu không đọc được khung hình

    current_plate = detect_plate(frame)  # Phát hiện biển số
    stabilize_plate(current_plate)  # Ổn định kết quả nhận diện

    # Chuyển đổi khung hình để hiển thị trên giao diện
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    frame_resized = cv2.resize(frame_rgb, (640, 480))   # Thay đổi kích thước để vừa giao diện
    img = Image.fromarray(frame_resized)  # Chuyển thành hình ảnh PIL
    imgtk = ImageTk.PhotoImage(image=img)  # Chuyển thành định dạng tương thích Tkinter

    camera_label.imgtk = imgtk  # Lưu tham chiếu để tránh bị thu hồi bộ nhớ
    camera_label.configure(image=imgtk)  # Cập nhật giao diện với khung hình mới

    camera_label.after(10, update_frame)  # Lên lịch cập nhật khung hình tiếp theo

# =====================================================================
# 5 GÁN SỰ KIỆN & VÒNG LẶP CHÍNH
# Gán các hàm cho nút điều khiển và chạy vòng lặp chính của Tkinter.
# =====================================================================
btn_start.config(command=start_camera)  # Gán hàm cho nút bắt đầu
btn_stop.config(command=stop_camera)    # Gán hàm cho nút dừng

window.mainloop()  # Chạy ứng dụng Tkinter

# =====================================================================
# 6 DỌN DẸP TÀI NGUYÊN
# Giải phóng tài nguyên camera và đóng tất cả cửa sổ OpenCV.
# =====================================================================
cap.release()
cv2.destroyAllWindows()