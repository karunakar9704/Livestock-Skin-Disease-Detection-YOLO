import zipfile
import os
zip_path = "/content/unzip1.zip"
extract_path = "/content/unzip1_dataset_extracted"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
zip_ref.extractall(extract_path)
print("Extraction complete!")
print(os.listdir(extract_path))
!ls /content/unzip1_dataset_extracted
# Point to your data.yaml file
data_yaml = "/content/unzip1_dataset_extracted/data.yaml"
import yaml
with open(f"{extract_path}/data.yaml", "r") as f:
data_cfg = yaml.safe_load(f)
print("Classes:", data_cfg["names"])
print("Number of classes:", data_cfg["nc"])
import os
root = "/content/unzip1_dataset_extracted"
for path, dirs, files in os.walk(root):
print(path, len(files))
pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.info()
results = model.train(
data=data_yaml,
epochs=150,
imgsz=65,
batch=16,
pretrained=True,
patience=10,
device=0
)
metrics = model.val()
print(metrics)
results = model.predict(source="/content/unzip1_dataset_extracted/test/images", 
save=True, imgsz=224)
import os
print("All run folders:\n", os.listdir("/content/runs/detect"))
import pandas as pd
results_csv = "/content/runs/detect/train/results.csv"
df = pd.read_csv(results_csv)
print(df.head()) # show first few rows
yolo detect predict model=/content/runs/detect/train/weights/best.pt 
source=/content/unzip1_dataset_extracted/valid/images
# Final epoch metrics
print("\nFinal Epoch Metrics:")
print(df.tail(1))