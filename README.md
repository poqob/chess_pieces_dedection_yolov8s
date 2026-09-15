# Chess Pieces Detection (YOLOv8s)

Bu proje, satranç taşlarını tespit etmek için **Ultralytics YOLOv8s** modeli ile hazırlanmış bir eğitim ve çıkarım çalışmasıdır.

## Proje İçeriği

- YOLOv8s ile model eğitimi
- Görseller üzerinde taş tespiti
- Video üzerinde gerçek zamanlıya yakın taş tespiti
- Eğitim çıktıları ve metriklerin kaydı (`runs/detect/...`)

## Veri Seti

Proje içinde Roboflow’dan dışa aktarılmış satranç taşları veri seti bulunur.

- Format: YOLOv8 (OBB/etiket dosyaları dahil)
- Sınıf sayısı: 13
- Sınıflar:
  - bishop
  - black-bishop, black-king, black-knight, black-pawn, black-queen, black-rook
  - white-bishop, white-king, white-knight, white-pawn, white-queen, white-rook

Önemli dosyalar:
- `/home/runner/work/chess_pieces_dedection_yolov8s/chess_pieces_dedection_yolov8s/datasets/Chess_pieces/data.yaml.yaml`
- `/home/runner/work/chess_pieces_dedection_yolov8s/chess_pieces_dedection_yolov8s/Chess Pieces.yolov8-obb/data.yaml`

## Klasör Yapısı

- `real-time-chess-pieces-detection-with-yolov8.ipynb`: Eğitim ve tahmin adımlarını içeren ana notebook
- `datasets/Chess_pieces/`: Eğitim/doğrulama/test verileri
- `runs/detect/`: Eğitim sonuçları, ağırlıklar ve metrikler
- `yolov8s.pt`: Başlangıç ağırlığı

## Kurulum

Python 3.10+ önerilir.

```bash
pip install ultralytics opencv-python matplotlib jupyter
```

## Kullanım

### 1) Notebook ile

Notebook dosyasını açın:

```bash
jupyter notebook "/home/runner/work/chess_pieces_dedection_yolov8s/chess_pieces_dedection_yolov8s/real-time-chess-pieces-detection-with-yolov8.ipynb"
```

Notebook içinde:
- Model yükleme (`YOLO('yolov8s.pt')`)
- Eğitim (`model.train(...)`)
- Görsel tahmini (`model.predict(source=...)`)
- Video tahmini (`model.predict(source=..., save=True)`)
adımları çalıştırılabilir.

### 2) Python kodu ile kısa örnek

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="datasets/Chess_pieces/data.yaml.yaml", epochs=15, batch=32, imgsz=640)
results = model.predict(source="datasets/Chess_pieces/Chess_video_example.mp4", conf=0.6, save=True)
```

## Eğitim Sonuçları (Mevcut Çalışmadan)

`runs/detect/train10/results.csv` çıktısına göre son epokta yaklaşık:
- Precision: **0.979**
- Recall: **0.989**
- mAP50: **0.991**
- mAP50-95: **0.781**

Ağırlık dosyaları:
- `runs/detect/train10/weights/best.pt`
- `runs/detect/train10/weights/last.pt`

## Notlar

- Veri yolları ortamınıza göre farklılık gösterebilir.
- Tahmin çıktıları varsayılan olarak `runs/detect/` altında saklanır.
- Daha iyi performans için `epochs`, `imgsz`, `batch` ve `conf` parametreleri ayarlanabilir.
