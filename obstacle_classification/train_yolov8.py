from ultralytics import YOLO

# 4.1 Choose your YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (extra-large)
# For training from scratch (random weights): use 'yolov8n.yaml', etc. 
# For transfer learning: use 'yolov8n.pt'

model = YOLO('yolov8n.pt')  # Recommended: transfer learning (faster convergence)
# model = YOLO('yolov8n.yaml')  # For true 'from scratch' training

# 4.2 Train on COCO
results = model.train(
    data='coco.yaml',      # path to your YAML
    epochs=100,            # typical epoch count
    imgsz=640,             # input image size (default 640)
    batch=16,              # tune for your GPU memory
    name='yolov8n_coco_exp_01'  # name of your experiment
)
