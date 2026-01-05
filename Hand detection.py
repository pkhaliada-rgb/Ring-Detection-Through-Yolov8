from ultralytics import YOLO
import os

# Initialize YOLOv8 model (choose size based on your needs)
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)
model = YOLO('yolov8n.pt')  # Start with nano for faster training

# Train the model
results = model.train(
    data='data.yaml',              # Path to your data.yaml file
    epochs=40,                    # Number of training epochs
    imgsz=640,                     # Image size (matches your preprocessing)
    batch=4,                       # Batch size (reduced for CPU training)
    device='cpu',                  # Use CPU (change to 0 for GPU if available)
    project='hand_ring_detection', # Project name
    name='yolov8n_run',           # Experiment name
    patience=50,                   # Early stopping patience
    save=True,                     # Save checkpoints
    plots=True,                    # Generate training plots
    
    # Advanced parameters (optional)
    lr0=0.01,                      # Initial learning rate
    lrf=0.01,                      # Final learning rate
    momentum=0.937,                # SGD momentum
    weight_decay=0.0005,           # Optimizer weight decay
    warmup_epochs=3.0,             # Warmup epochs
    amp=True,                      # Automatic Mixed Precision
    
    # Data augmentation (already applied in your dataset)
    hsv_h=0.015,                   # HSV-Hue augmentation
    hsv_s=0.7,                     # HSV-Saturation augmentation
    hsv_v=0.4,                     # HSV-Value augmentation
    degrees=0.0,                   # Rotation (already applied in preprocessing)
    translate=0.1,                 # Translation
    scale=0.5,                     # Scale
    flipud=0.0,                    # Flip up-down probability
    fliplr=0.5,                    # Flip left-right probability (matches your preprocessing)
)

# Validate the model
metrics = model.val()

# Print validation metrics
print("\n=== Validation Metrics ===")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Export the model (optional)
# model.export(format='onnx')  # Export to ONNX format
# model.export(format='torchscript')  # Export to TorchScript