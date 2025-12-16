import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, 
                             QMessageBox, QSlider, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QFrame, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

# --- MODERN STİL ---
STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QLabel { color: #ffffff; font-size: 14px; }
QPushButton {
    background-color: #007acc; color: white; border-radius: 5px;
    padding: 10px; font-weight: bold; font-size: 14px;
}
QPushButton:hover { background-color: #005f9e; }
QPushButton:pressed { background-color: #003f6b; }
QTableWidget {
    background-color: #3c3c3c; color: white; gridline-color: #555;
    border: none; border-radius: 5px;
}
QHeaderView::section {
    background-color: #444; color: white; padding: 5px; border: 1px solid #555;
}
"""

class YOLOv8_ONNX:
    def __init__(self, model_path):
        self.classes = ['Kesici_Delici', 'Tasiyici_Servis'] 
        self.net = None
        self.error_message = None
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            if self.net.empty():
                raise Exception("Model dosyası boş veya geçersiz")
        except Exception as e:
            self.error_message = str(e)
            self.net = None

    def detect(self, image, conf_threshold=0.5):
        if self.net is None: return []

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= conf_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], 
                    outputs[0][i][3]
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.3)
        detections = []
        if len(result_boxes) > 0:
            for index in result_boxes.flatten():
                detections.append({
                    "class_id": class_ids[index],
                    "class_name": self.classes[class_ids[index]] if class_ids[index] < len(self.classes) else "Unknown",
                    "confidence": scores[index],
                    "box": boxes[index]
                })
        return detections

class ProMutfakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Nesne Tespiti Pro")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLESHEET)
        
        self.current_model_path = "best.onnx"
        self.detector = YOLOv8_ONNX(self.current_model_path) 
        self.current_image = None
        self.camera = None
        self.camera_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.fixed_frame_size = (640, 480)
        self.colors = {'Kesici_Delici': (0, 165, 255), 'Tasiyici_Servis': (0, 255, 127), 'Unknown': (255, 255, 255)}
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- SOL PANEL ---
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("background-color: #333; border-radius: 10px;")
        left_layout = QVBoxLayout(left_panel)

        self.btn_load = QPushButton("📁 Resim Yükle")
        self.btn_load.clicked.connect(self.select_image)
        left_layout.addWidget(self.btn_load)

        self.btn_camera = QPushButton("📹 Test Camera")
        self.btn_camera.clicked.connect(self.toggle_camera)
        left_layout.addWidget(self.btn_camera)

        left_layout.addSpacing(10)
        self.lbl_model = QLabel("Model Seç:")
        left_layout.addWidget(self.lbl_model)
        
        self.combo_model = QComboBox()
        self.combo_model.addItem("best.onnx", "best.onnx")
        if os.path.exists("örnek/best.onnx"):
            self.combo_model.addItem("örnek/best.onnx", "örnek/best.onnx")
        self.combo_model.addItem("📁 Özel Model Seç...", "")
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)
        self.combo_model.setStyleSheet("""
            QComboBox {
                background-color: #444; color: white; border-radius: 5px;
                padding: 5px; font-size: 13px;
            }
            QComboBox:hover { background-color: #555; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #444; color: white;
                selection-background-color: #007acc;
            }
        """)
        left_layout.addWidget(self.combo_model)

        left_layout.addSpacing(10)
        self.lbl_conf = QLabel("Güven Eşiği: %50")
        left_layout.addWidget(self.lbl_conf)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(95)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(lambda: self.lbl_conf.setText(f"Güven Eşiği: %{self.slider.value()}"))
        self.slider.sliderReleased.connect(self.run_detection)
        left_layout.addWidget(self.slider)

        left_layout.addSpacing(10)
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Nesne", "Güven %"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)
        main_layout.addWidget(left_panel)

        # --- SAĞ PANEL ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.image_label = QLabel("Resim Bekleniyor...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #444; background-color: #222;")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(False)
        right_layout.addWidget(self.image_label)
        main_layout.addWidget(right_panel)

    def on_model_changed(self, index):
        """Model değiştiğinde çağrılır"""
        model_path = self.combo_model.itemData(index)
        
        # Eğer "Özel Model Seç" seçildiyse
        if model_path == "":
            path, _ = QFileDialog.getOpenFileName(
                self, "ONNX Model Dosyası Seç", "", 
                "ONNX Dosyaları (*.onnx);;Tüm Dosyalar (*.*)"
            )
            if path:
                # Yeni modeli combo'ya ekle
                filename = os.path.basename(path)
                self.combo_model.blockSignals(True)
                self.combo_model.insertItem(self.combo_model.count() - 1, filename, path)
                self.combo_model.setCurrentIndex(self.combo_model.count() - 2)
                self.combo_model.blockSignals(False)
                self.load_model(path)
            else:
                # İptal edildiyse önceki modele geri dön
                self._restore_model_selection(self.current_model_path)
        else:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Yeni modeli yükle"""
        old_model_path = self.current_model_path
        
        if self.camera_active:
            QMessageBox.warning(
                self, "Uyarı", 
                "Model değiştirmek için önce kamerayı durdurun."
            )
            self._restore_model_selection(old_model_path)
            return
        
        try:
            self.detector = YOLOv8_ONNX(model_path)
            if self.detector.net is None:
                error_msg = self.detector.error_message if self.detector.error_message else "Model yüklenemedi"
                raise Exception(error_msg)
            
            # Model başarıyla yüklendi, current_model_path'i güncelle
            self.current_model_path = model_path
            
            QMessageBox.information(
                self, "Başarılı", 
                f"Model başarıyla yüklendi:\n{os.path.basename(model_path)}"
            )
            
            # Eğer bir resim yüklüyse, tespiti yeniden çalıştır
            if self.current_image is not None:
                self.run_detection()
        except Exception as e:
            QMessageBox.critical(
                self, "Hata", 
                f"Model yüklenirken hata oluştu:\n\n{str(e)}\n\nEski model kullanılmaya devam edilecek."
            )
            self._restore_model_selection(old_model_path)
            # Eski modeli tekrar yükle (eğer hala geçerliyse)
            if os.path.exists(old_model_path):
                try:
                    self.detector = YOLOv8_ONNX(old_model_path)
                except:
                    pass

    def _restore_model_selection(self, model_path):
        """Combo box'ta belirtilen modeli seç"""
        self.combo_model.blockSignals(True)
        for i in range(self.combo_model.count()):
            if self.combo_model.itemData(i) == model_path:
                self.combo_model.setCurrentIndex(i)
                break
        self.combo_model.blockSignals(False)

    def _draw_detections(self, frame, detections, scale_x=1.0, scale_y=1.0, min_size=30):
        """Tespitleri çiz ve tabloya ekle"""
        w = frame.shape[1]
        self.table.setRowCount(0)

        for det in detections:
            box = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            w_box = int(box[2] * scale_x)
            h_box = int(box[3] * scale_y)

            if w_box < min_size or h_box < min_size:
                continue

            color = self.colors.get(cls_name, (255, 0, 0))
            cv2.rectangle(frame, (x1, y1), (x1 + w_box, y1 + h_box), color, 1)

            label_text = f"{cls_name} {int(conf*100)}%"
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5 if scale_x == 1.0 else 0.40
            thickness = 1

            (tw, th), _ = cv2.getTextSize(label_text, font_face, font_scale, thickness)
            if tw > w_box:
                font_scale = w_box / (tw * 1.3)
                (tw, th), _ = cv2.getTextSize(label_text, font_face, font_scale, thickness)

            box_w, box_h = tw + 6, th + 6
            label_x = max(0, min(x1, w - box_w - 2))

            if y1 - box_h < 0:
                rect_y1, rect_y2 = y1, y1 + box_h
                text_y = rect_y1 + th + 2
            else:
                rect_y1, rect_y2 = y1 - box_h, y1
                text_y = rect_y2 - 3

            bg_color = color if scale_x == 1.0 else (0, 0, 0)
            cv2.rectangle(frame, (label_x, rect_y1), (label_x + box_w, rect_y2), bg_color, -1)
            if scale_x == 1.0:
                cv2.rectangle(frame, (label_x, rect_y1), (label_x + box_w, rect_y2), (255, 255, 255), 1)
            cv2.putText(frame, label_text, (label_x + 2, text_y),
                       font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(cls_name))
            self.table.setItem(row, 1, QTableWidgetItem(f"%{int(conf*100)}"))

    def _frame_to_pixmap(self, frame, target_size=None):
        """OpenCV frame'ini QPixmap'e dönüştür"""
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if target_size:
            return pixmap.scaled(target_size[0], target_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def select_image(self):
        if self.camera_active:
            self.toggle_camera()
        
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.jpg *.png *.jpeg)")
        if path:
            self.current_image = cv2.imread(path)
            self.run_detection()

    def toggle_camera(self):
        """Kamerayı aç/kapat"""
        if not self.camera_active:
            try:
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except:
                self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                QMessageBox.warning(self, "Hata", "Kamera açılamadı!\n\nLütfen:\n- Kameranın bağlı olduğundan emin olun\n- Başka bir program kamerayı kullanmıyor olmalı")
                return
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.fixed_frame_size[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.fixed_frame_size[1])
            
            if not self.camera.read()[0]:
                QMessageBox.warning(self, "Hata", "Kamera başlatılamadı!")
                self.camera.release()
                self.camera = None
                return
            
            self.camera_active = True
            self.btn_camera.setText("⏹️ Durdur")
            self.btn_load.setEnabled(False)
            self.timer.start(33)
        else:
            self.camera_active = False
            self.timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            self.btn_camera.setText("📹 Test Camera")
            self.btn_load.setEnabled(True)
            self.table.setRowCount(0)

    def update_camera_frame(self):
        """Kamera frame'lerini yakala ve işle"""
        if not self.camera_active or not self.camera:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            return
        
        frame = cv2.resize(frame, self.fixed_frame_size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        detections = self.detector.detect(frame, conf_threshold=self.slider.value() / 100.0)
        self._draw_detections(frame, detections)
        self.image_label.setPixmap(self._frame_to_pixmap(frame, self.fixed_frame_size))

    def run_detection(self):
        if self.camera_active or self.current_image is None:
            return
        
        img_copy = self.current_image.copy()
        h, w = img_copy.shape[:2]
        detections = self.detector.detect(img_copy, conf_threshold=self.slider.value() / 100.0)
        self._draw_detections(img_copy, detections, w / 640, h / 640)
        self.image_label.setPixmap(self._frame_to_pixmap(img_copy))

    def closeEvent(self, event):
        if self.camera_active:
            self.toggle_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProMutfakApp()
    window.show()
    sys.exit(app.exec_())
