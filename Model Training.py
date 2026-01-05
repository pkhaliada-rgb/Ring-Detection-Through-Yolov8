from ultralytics import YOLO
import os
import yaml

DATASET_PATH = r"C:\Users\silicon\Downloads\Ring Area Detection.v6i.yolov8"
PROJECT_NAME = "ring_detection"
RUN_NAME = "yolov8_train"

MODEL_SIZE = 'yolov8n.pt'
EPOCHS = 30
IMAGE_SIZE = 416
BATCH_SIZE = 8
DEVICE = 'cpu'
PATIENCE = 50
OPTIMIZER = 'auto'
LEARNING_RATE = 0.01
AUGMENT = True

def check_dataset_structure(dataset_path):
    print("=" * 50)
    print("Checking Dataset Structure...")
    print("=" * 50)
    
    required_items = ['data.yaml', 'train', 'valid']
    
    for item in required_items:
        item_path = os.path.join(dataset_path, item)
        if os.path.exists(item_path):
            print(f"✓ Found: {item}")
            
            if os.path.isdir(item_path):
                images_path = os.path.join(item_path, 'images')
                labels_path = os.path.join(item_path, 'labels')
                
                if os.path.exists(images_path):
                    num_images = len([f for f in os.listdir(images_path) 
                                    if f.endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  - Images: {num_images}")
                
                if os.path.exists(labels_path):
                    num_labels = len([f for f in os.listdir(labels_path) 
                                    if f.endswith('.txt')])
                    print(f"  - Labels: {num_labels}")
        else:
            print(f"✗ Missing: {item}")
    
    print("=" * 50)

def load_data_config(dataset_path):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\nDataset Configuration:")
        print(f"  - Number of classes: {config.get('nc', 'Not specified')}")
        print(f"  - Class names: {config.get('names', 'Not specified')}")
        print(f"  - Train path: {config.get('train', 'Not specified')}")
        print(f"  - Validation path: {config.get('val', 'Not specified')}")
        
        return yaml_path
    else:
        print("⚠ Warning: data.yaml not found!")
        return None

def train_yolov8(data_yaml_path):
    print("\n" + "=" * 50)
    print("Starting YOLOv8 Training...")
    print("=" * 50)
    
    model = YOLO(MODEL_SIZE)
    
    print(f"\nModel: {MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {'GPU' if DEVICE == 0 else 'CPU'}")
    
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        augment=AUGMENT,
        plots=True,
        save=True,
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    
    return model, results

def validate_model(model, data_yaml_path):
    print("\n" + "=" * 50)
    print("Validating Model...")
    print("=" * 50)
    
    metrics = model.val(data=data_yaml_path)
    
    print(f"\nValidation Results:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    return metrics

def export_model(model):
    print("\n" + "=" * 50)
    print("Exporting Model...")
    print("=" * 50)
    
    model.export(format='onnx')
    print("✓ Exported to ONNX format")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("YOLOv8 RING AREA DETECTION TRAINING")
    print("=" * 50)
    
    check_dataset_structure(DATASET_PATH)
    
    data_yaml = load_data_config(DATASET_PATH)
    
    if data_yaml is None:
        print("\n⚠ Error: Cannot proceed without data.yaml file!")
        exit(1)
    
    model, results = train_yolov8(data_yaml)
    
    metrics = validate_model(model, data_yaml)
    
    export_model(model)
    
    print("\n" + "=" * 50)
    print("ALL DONE! 🎉")
    print("=" * 50)
    print(f"\nTrained model saved in: {PROJECT_NAME}/{RUN_NAME}/")
    print("Check the 'weights' folder for best.pt and last.pt")