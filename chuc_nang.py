import cv2
from ultralytics import YOLO
import easyocr
import datetime
import random
from bien_so_map_dau import BIEN_SO_MAP_DAU   # Bản đồ mã tỉnh → tên địa phương (đầu biển số)
from bien_so_map import BIEN_SO_MAP           # Bản đồ mã biển số → địa phương
from chu_xe import CHU_XE                     # Danh sách tên chủ xe ngẫu nhiên

# =====================================================================
# CẤU HÌNH MÔ HÌNH & OCR
# ---------------------------------------------------------------------
# - Sử dụng mô hình YOLO để phát hiện vùng chứa biển số xe trong hình.
# - Sử dụng EasyOCR để nhận diện ký tự trên biển số (text recognition).
# =====================================================================

MODEL_PATH = "D:/Do_An_AI/runs/detect/train_bien_so_100epoch/weights/best.pt"
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['vi'], gpu=True) # thay False nếu không có cài đặt cuda và GPU mạnh

# =====================================================================
# LỚP PlateRecognizer — XỬ LÝ NHẬN DIỆN BIỂN SỐ
# ---------------------------------------------------------------------
# Chức năng:
#   - Mở và đọc luồng camera (Webcam)
#   - Phát hiện vị trí biển số bằng YOLO
#   - Nhận diện ký tự bằng EasyOCR
#   - Xác nhận biển số ổn định (lọc nhiễu)
#   - Gắn thông tin chủ xe và địa phương
# =====================================================================
class PlateRecognizer:
    def __init__(self):
        """Khởi tạo camera và các biến dùng trong quá trình nhận diện."""
        # Mở camera (ID 0 = camera mặc định)
        self.cap = cv2.VideoCapture(0)
        # Đặt kích thước khung hình camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Trạng thái hoạt động của camera
        self.running = False
        # Đếm số khung hình (để giảm tần suất YOLO chạy)
        self.frame_count = 0
        # Lưu kết quả YOLO gần nhất (để dùng lại giữa các khung hình)
        self.last_results = []
        # Bộ đệm lưu tạm biển số (để xác nhận biển số ổn định)
        self.plate_buffer = []
        # Biển số đã được xác nhận gần nhất
        self.last_confirmed_plate = ""
        # Bản đồ lưu thông tin chủ xe cho từng biển số
        self.plate_owner_map = {}

    # -----------------------------------------------------------------
    # PHÁT HIỆN BIỂN SỐ TRONG KHUNG HÌNH
    # -----------------------------------------------------------------
    def detect_plate(self, frame):
        """
        Hàm phát hiện biển số trong khung hình.
        - Dùng YOLO để xác định vị trí biển số.
        - Dùng EasyOCR để đọc ký tự trong vùng biển số.
        - Trả về chuỗi ký tự của biển số hiện tại.
        """
        self.frame_count += 1

        # Để tránh chạy YOLO ở mỗi khung hình (tốn tài nguyên),
        # chỉ chạy mỗi 15 khung hình một lần.
        if self.frame_count % 15 == 0:
            results = model(frame)           # Phát hiện mới bằng YOLO
            self.last_results = results      # Lưu lại kết quả mới
        else:
            results = self.last_results      # Dùng kết quả trước đó

        current_plate = ""  # Biển số hiện tại (đọc được trong khung này)

        # Duyệt qua các kết quả phát hiện của YOLO
        for r in results:
            # Lấy danh sách tọa độ hộp (bounding boxes)
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ khung
                # Vẽ khung quanh biển số
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Cắt vùng chứa biển số ra khỏi khung hình
                bien_so_crop = frame[y1:y2, x1:x2]
                if bien_so_crop.size == 0:
                    continue  # Nếu ảnh rỗng thì bỏ qua

                # Dùng OCR đọc chữ trên biển số
                result = reader.readtext(bien_so_crop)
                # Ghép các chuỗi ký tự đọc được
                text = " ".join([res[1] for res in result]) if result else ""

                if text:
                    # Lấy mã tỉnh từ biển số (VD: “51F-123.45” → “51”)
                    ma_tinh = text.split("-")[0] if "-" in text else text[:2]
                    # Tra cứu địa phương tương ứng
                    dia_phuong = BIEN_SO_MAP.get(ma_tinh, "Không rõ địa phương")
                    current_plate = text
                    # Hiển thị biển số + địa phương lên khung hình
                    cv2.putText(
                        frame,
                        f"{text} ({dia_phuong})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

        return current_plate  # Trả về biển số đọc được

    # -----------------------------------------------------------------
    # ỔN ĐỊNH KẾT QUẢ NHẬN DIỆN (CHỐNG NHIỄU)
    # -----------------------------------------------------------------
    def stabilize_plate(self, current_plate):
        """
        Kiểm tra xem biển số có xuất hiện liên tục đủ lâu
        để xác nhận là hợp lệ hay không.
        Khi đủ điều kiện, trả về thông tin chi tiết:
            - Biển số
            - Tên chủ xe
            - Địa phương
            - Thời gian nhận diện
        """
        if not current_plate:
            return None  # Nếu không có biển số, bỏ qua

        # Thêm biển số hiện tại vào bộ đệm
        self.plate_buffer.append(current_plate)
        # Giới hạn bộ đệm 5 phần tử (FIFO)
        if len(self.plate_buffer) > 5:
            self.plate_buffer.pop(0)

        # Nếu cùng 1 biển số xuất hiện >= 3 lần trong 5 khung gần nhất
        # và khác với biển số đã xác nhận gần nhất → xác nhận mới
        if self.plate_buffer.count(current_plate) >= 3 and current_plate != self.last_confirmed_plate:
            # Lấy mã tỉnh từ biển số
            ma_tinh = current_plate.split("-")[0] if "-" in current_plate else current_plate[:2]
            # Tra cứu tên địa phương từ mã tỉnh
            dia_phuong = BIEN_SO_MAP_DAU.get(ma_tinh, "Không rõ địa phương")

            # Nếu biển số chưa có chủ xe → gán ngẫu nhiên
            if current_plate not in self.plate_owner_map:
                self.plate_owner_map[current_plate] = random.choice(CHU_XE)
            ten_chu_xe = self.plate_owner_map[current_plate]

            # Lấy thời gian hiện tại
            now = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            # Cập nhật biển số đã xác nhận gần nhất
            self.last_confirmed_plate = current_plate

            # Trả về kết quả nhận diện ổn định
            return {
                "plate": current_plate,
                "owner": ten_chu_xe,
                "location": dia_phuong,
                "time": now
            }

        return None  # Nếu chưa đủ điều kiện ổn định thì không trả về gì

    # -----------------------------------------------------------------
    # GIẢI PHÓNG TÀI NGUYÊN
    # -----------------------------------------------------------------
    def release(self):
        """Giải phóng camera và đóng các cửa sổ OpenCV."""
        self.cap.release()
        cv2.destroyAllWindows()
