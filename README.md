# 🍴 YOLOv8 Mutfak Eşyaları Nesne Tespiti Projesi

Bu proje, mutfak eşyalarını (Kesici_Delici ve Tasiyici_Servis) tespit etmek için YOLOv8 modeli eğiten ve PyQt5 tabanlı görsel arayüz uygulaması içeren kapsamlı bir nesne tespiti sistemidir.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Model Eğitimi (Google Colab)](#model-eğitimi-google-colab)
- [GUI Uygulaması](#gui-uygulaması)
- [Teknik Detaylar](#teknik-detaylar)
- [Sorun Giderme](#sorun-giderme)

## 🎯 Proje Hakkında

Bu proje, BLG-407 Makine Öğrenmesi dersi kapsamında geliştirilmiştir. Proje iki ana bileşenden oluşmaktadır:

1. **Model Eğitimi**: Google Colab üzerinde YOLOv8 modeli eğitilir ve ONNX formatına dönüştürülür
2. **GUI Uygulaması**: PyQt5 ile geliştirilmiş, gerçek zamanlı nesne tespiti yapabilen masaüstü uygulaması

### Tespit Edilen Sınıflar

- **Kesici_Delici**: Bıçak, makas gibi kesici ve delici mutfak eşyaları
- **Tasiyici_Servis**: Taba, servis tabağı gibi taşıyıcı ve servis eşyaları

## ✨ Özellikler

### Model Eğitimi
- ✅ Roboflow platformundan veri seti indirme
- ✅ YOLOv8l (large) modeli ile eğitim
- ✅ Detaylı performans metrikleri (mAP50, mAP50-95, Precision, Recall)
- ✅ Otomatik ONNX formatına dönüştürme
- ✅ Eğitim grafikleri ve karmaşıklık matrisi

### GUI Uygulaması
- ✅ **Resim Yükleme**: JPG, PNG, JPEG formatlarında resim yükleme
- ✅ **Webcam Desteği**: Gerçek zamanlı kamera ile nesne tespiti
- ✅ **Model Seçimi**: Farklı ONNX modelleri arasında seçim yapma
- ✅ **Güven Eşiği Ayarı**: Slider ile dinamik güven eşiği ayarlama (10%-95%)
- ✅ **Detaylı Sonuçlar**: Tespit edilen nesnelerin listesi ve güven skorları
- ✅ **Modern Arayüz**: Koyu tema ile modern ve kullanıcı dostu tasarım

## 📦 Gereksinimler

### Model Eğitimi için (Google Colab)
- Google Colab hesabı
- Google Drive erişimi
- Roboflow API anahtarı

### GUI Uygulaması için (Yerel Ortam)
- Python 3.7 veya üzeri
- PyQt5
- OpenCV (cv2)
- NumPy

## 🚀 Kurulum

### 1. Projeyi İndirin

```bash
git clone https://github.com/kullanici_adi/YoloV8_Nesne_Tespiti
cd YoloV8_Nesne_Tespiti
```

### 2. GUI Uygulaması için Gerekli Kütüphaneleri Kurun

```bash
pip install PyQt5 opencv-python numpy
```

veya `requirements.txt` dosyası kullanarak:

```bash
pip install -r requirements.txt
```

### 3. Model Dosyalarını Hazırlayın

- `best.onnx` dosyasının proje klasöründe olduğundan emin olun
- Eğer farklı bir model kullanmak istiyorsanız, `main.py` dosyasındaki `current_model_path` değişkenini güncelleyin

## 📖 Kullanım

### GUI Uygulamasını Çalıştırma

```bash
python main.py
```

### Uygulama Özellikleri

1. **Resim Yükleme**:
   - "📁 Resim Yükle" butonuna tıklayın
   - Bir görüntü dosyası seçin (JPG, PNG, JPEG)
   - Tespit sonuçları otomatik olarak gösterilir

2. **Webcam Kullanımı**:
   - "📹 Test Camera" butonuna tıklayın
   - Kameranız açılır ve gerçek zamanlı tespit başlar
   - "⏹️ Durdur" butonuna tıklayarak kamerayı kapatabilirsiniz

3. **Model Seçimi**:
   - "Model Seç" dropdown menüsünden farklı modeller seçebilirsiniz
   - "📁 Özel Model Seç..." seçeneği ile kendi ONNX modelinizi yükleyebilirsiniz

4. **Güven Eşiği Ayarı**:
   - Slider'ı kullanarak güven eşiğini ayarlayın (10%-95%)
   - Daha yüksek değerler daha az ama daha güvenilir tespitler verir
   - Daha düşük değerler daha fazla ama daha az güvenilir tespitler verir

## 📁 Proje Yapısı

```
Mutfak/
│
├── odev2.ipynb              # Google Colab'da çalıştırılacak model eğitimi notebook'u
├── main.py                   # PyQt5 GUI uygulaması
├── best.pt                   # Eğitilmiş PyTorch model dosyası
├── best.onnx                 # ONNX formatına dönüştürülmüş model dosyası
├── README.md                 # Bu dosya
└── requirements.txt          # Python bağımlılıkları (opsiyonel)
```

## 🎓 Model Eğitimi (Google Colab)

### Adım 1: Notebook'u Açın

1. Google Colab'da `odev2.ipynb` dosyasını açın
2. Runtime → Change runtime type → GPU seçin (önerilir)

### Adım 2: Google Drive'ı Bağlayın

Notebook'un ilk hücresinde Google Drive otomatik olarak bağlanır:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Adım 3: API Anahtarını Hazırlayın

1. Google Drive'ınızda şu klasör yapısını oluşturun:
   ```
   /content/drive/MyDrive/makine öğrenmesi/ödev2/API_Keys/
   ```

2. `roboflow_api_key.txt` dosyası oluşturun ve içine Roboflow API anahtarınızı yazın

3. Notebook'taki `api_key_path` değişkenini kendi yolunuza göre güncelleyin

### Adım 4: Veri Setini İndirin

Notebook, Roboflow platformundan veri setini otomatik olarak indirir:

```python
rf = Roboflow(api_key=api_key)
project = rf.workspace("erdem-5ubfq").project("mutfak-wvkhr")
version = project.version(2)
dataset = version.download("yolov8")
```

### Adım 5: Modeli Eğitin

```python
model = YOLO('yolov8l.pt')
results = model.train(
    data=f"{dataset_path}/data.yaml",
    epochs=100,
    imgsz=640,
    name='mutfak_esya_modeli',
    patience=50
)
```

### Adım 6: Modeli ONNX Formatına Dönüştürün

```python
model.export(format='onnx')
```

### Adım 7: Model Dosyalarını İndirin

Eğitim tamamlandıktan sonra:
- `best.pt` dosyasını indirin
- `best.onnx` dosyasını indirin
- Bu dosyaları yerel proje klasörünüze kopyalayın

## 🖥️ GUI Uygulaması

### Özellikler

- **Modern Koyu Tema**: Göz yormayan koyu renk şeması
- **Gerçek Zamanlı İşleme**: Webcam ile canlı nesne tespiti
- **Dinamik Model Yükleme**: Çalışma zamanında farklı modeller yüklenebilir
- **Detaylı Sonuç Tablosu**: Tespit edilen her nesne için sınıf ve güven skoru

### Sistem Gereksinimleri

- **İşletim Sistemi**: Windows, Linux, macOS
- **RAM**: Minimum 4GB (önerilen 8GB)
- **Kamera**: Webcam desteği (opsiyonel)
- **Python**: 3.7 veya üzeri

## 🔧 Teknik Detaylar

### Model Mimarisi

- **Temel Model**: YOLOv8l (Large)
- **Giriş Boyutu**: 640x640 piksel
- **Çıkış Formatı**: ONNX (Open Neural Network Exchange)
- **Sınıf Sayısı**: 2 (Kesici_Delici, Tasiyici_Servis)

### Eğitim Parametreleri

- **Epochs**: 100
- **Batch Size**: 16 (otomatik)
- **Image Size**: 640x640
- **Patience**: 50 (early stopping)
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (başlangıç)

### Performans Metrikleri

Model eğitimi sonrası elde edilen metrikler:

- **mAP50**: ~98.36%
- **mAP50-95**: ~70.50%
- **Precision**: ~100.00%
- **Recall**: ~89.32%
- **F1-Score**: ~94.36%

### GUI Teknolojileri

- **Framework**: PyQt5
- **Görüntü İşleme**: OpenCV (cv2)
- **Model Inference**: OpenCV DNN (ONNX Runtime)
- **Görüntü Formatı**: BGR → RGB dönüşümü

## 🐛 Sorun Giderme

### Model Yüklenmiyor

- `best.onnx` dosyasının proje klasöründe olduğundan emin olun
- Dosya yolunu kontrol edin
- Model dosyasının bozuk olmadığından emin olun

### Kamera Açılmıyor

- Kameranın başka bir program tarafından kullanılmadığından emin olun
- Kamera sürücülerinin güncel olduğundan emin olun
- Windows'ta DirectShow backend kullanılıyor (otomatik)

### Tespit Yapılmıyor

- Güven eşiğini düşürün (slider'ı sola kaydırın)
- Işık koşullarını kontrol edin
- Modelin eğitildiği nesnelere benzer nesneler kullanın

### Colab'da Hata Alıyorum

- GPU runtime'ın aktif olduğundan emin olun
- API anahtarının doğru olduğundan emin olun
- Google Drive'ın düzgün bağlandığından emin olun

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👤 Geliştirici

**Mustafa Erdem Kaya**
- Okul Numarası: 2212721009
- GitHub: [kullanici_adi](https://github.com/erdemkaya4332)

## 🙏 Teşekkürler

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 implementasyonu
- [Roboflow](https://roboflow.com/) - Veri seti platformu
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework

## 📚 Kaynaklar

- [YOLOv8 Dokümantasyonu](https://docs.ultralytics.com/)
- [Roboflow Dokümantasyonu](https://docs.roboflow.com/)
- [PyQt5 Dokümantasyonu](https://www.riverbankcomputing.com/static/Docs/PyQt5/)

---

**Not**: Bu proje BLG-407 Makine Öğrenmesi dersi kapsamında geliştirilmiştir.

