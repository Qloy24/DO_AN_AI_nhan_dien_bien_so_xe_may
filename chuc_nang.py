import cv2
from ultralytics import YOLO
import easyocr
import datetime
import random
from bien_so_map_dau import BIEN_SO_MAP_DAU
from bien_so_map import BIEN_SO_MAP
from chu_xe import CHU_XE

MODEL_PATH = "D:/Do_An_AI/runs/detect/train_bien_so_100epoch/weights/best.pt"
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['vi'], gpu=True)

class PlateRecognizer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = False
        self.frame_count = 0
        self.last_results = []
        self.plate_buffer = []
        self.last_confirmed_plate = ""
        self.plate_owner_map = {}

    def detect_plate(self, frame):
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            results = model(frame)
            self.last_results = results
        else:
            results = self.last_results

        current_plate = ""

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                bien_so_crop = frame[y1:y2, x1:x2]
                if bien_so_crop.size == 0:
                    continue

                result = reader.readtext(bien_so_crop)
                text = " ".join([res[1] for res in result]) if result else ""

                if text:
                    ma_tinh = text.split("-")[0] if "-" in text else text[:2]
                    dia_phuong = BIEN_SO_MAP.get(ma_tinh, "Không rõ địa phương")
                    current_plate = text
                    cv2.putText(frame, f"{text} ({dia_phuong})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return current_plate

    def stabilize_plate(self, current_plate):
        if not current_plate:
            return None

        self.plate_buffer.append(current_plate)
        if len(self.plate_buffer) > 5:
            self.plate_buffer.pop(0)

        if self.plate_buffer.count(current_plate) >= 3 and current_plate != self.last_confirmed_plate:
            ma_tinh = current_plate.split("-")[0] if "-" in current_plate else current_plate[:2]
            dia_phuong = BIEN_SO_MAP_DAU.get(ma_tinh, "Không rõ địa phương")

            if current_plate not in self.plate_owner_map:
                self.plate_owner_map[current_plate] = random.choice(CHU_XE)
            ten_chu_xe = self.plate_owner_map[current_plate]

            now = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            self.last_confirmed_plate = current_plate

            return {
                "plate": current_plate,
                "owner": ten_chu_xe,
                "location": dia_phuong,
                "time": now
            }

        return None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
