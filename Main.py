from chuc_nang import PlateRecognizer
from giao_dien import PlateUI

if __name__ == "__main__":
    recognizer = PlateRecognizer()
    app = PlateUI(recognizer)
    app.run()