from ultralytics import YOLO
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_yolo_model(model_path, data_yaml):
    """
    Comprehensive evaluation of YOLO model with human-readable metrics
    """
    print("="*50)
    print("YOLO MODEL EVALUATION REPORT")
    print("="*50)
    
    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Model info
    print(f"Model: {model_path}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Time the validation
    print("\nRunning validation...")
    start_time = time.time()
    
    # Run validation
    metrics = model.val(data=data_yaml, verbose=False)
    
    end_time = time.time()
    validation_time = end_time - start_time
    
    # Extract key metrics
    results = metrics.box  # Detection metrics
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    
    # Accuracy Metrics
    print(f"ACCURACY METRICS:")
    print(f"   mAP@0.5      : {results.map50:.4f} ({results.map50*100:.2f}%)")
    print(f"   mAP@0.5:0.95 : {results.map:.4f} ({results.map*100:.2f}%)")
    
    # Per-class metrics (assuming single class - pothole)
    if len(results.ap) > 0:
        print(f"   AP@0.5       : {results.ap[0]:.4f} ({results.ap[0]*100:.2f}%)")
    
    print(f"\n DETECTION METRICS:")
    print(f"   Precision    : {results.p[0]:.4f} ({results.p[0]*100:.2f}%)")
    print(f"   Recall       : {results.r[0]:.4f} ({results.r[0]*100:.2f}%)")
    
    # Calculate F1 Score
    if results.p[0] + results.r[0] > 0:
        f1_score = 2 * (results.p[0] * results.r[0]) / (results.p[0] + results.r[0])
        print(f"   F1-Score     : {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    print(f"\n PERFORMANCE METRICS:")
    print(f"   Validation Time: {validation_time:.2f} seconds")
    
    # Speed metrics (if available)
    if hasattr(results, 'speed'):
        speed = results.speed
        print(f"   Inference Speed: {speed['inference']:.1f} ms/image")
        print(f"   Postprocess    : {speed['postprocess']:.1f} ms/image")
        print(f"   Total Speed    : {speed['inference'] + speed['postprocess']:.1f} ms/image")
        
        # Calculate FPS
        total_speed_seconds = (speed['inference'] + speed['postprocess']) / 1000
        fps = 1 / total_speed_seconds if total_speed_seconds > 0 else 0
        print(f"   FPS            : {fps:.1f} frames/second")
    
    # Model interpretation
    print(f"\n MODEL PERFORMANCE INTERPRETATION:")
    
    map50 = results.map50
    if map50 >= 0.9:
        performance = "EXCELLENT"
    elif map50 >= 0.8:
        performance = " GOOD"
    elif map50 >= 0.6:
        performance = " FAIR"
    else:
        performance = "NEEDS IMPROVEMENT"
    
    print(f"   Overall Performance: {performance}")
    print(f"   mAP@0.5 = {map50:.3f} means  model correctly detects {map50*100:.1f}% of potholes")
    
    precision = results.p[0]
    recall = results.r[0]
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"   • Precision ({precision:.3f}): Out of all detections, {precision*100:.1f}% are actually potholes")
    print(f"   • Recall ({recall:.3f}): Out of all actual potholes, {recall*100:.1f}% are detected")
    
    if precision > recall:
        print(f"    model is conservative - fewer false positives, but might miss some potholes")
    elif recall > precision:
        print(f"    model is aggressive - catches most potholes, but has some false detections")
    else:
        print(f"    model has balanced precision and recall")
    
    # Recommendations
    print(f"\n RECOMMENDATIONS:")
    if map50 < 0.7:
        print(f"   • Consider training for more epochs")
        print(f"   • Check if dataset has sufficient variety")
        print(f"   • Verify label quality")
    elif map50 < 0.9:
        print(f"   • Model performance is good, minor tuning may help")
        print(f"   • Consider data augmentation")
    else:
        print(f"   • Excellent performance! Model is ready for deployment")
    
    print("="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    return metrics

# Usage
if __name__ == "__main__":
    
    model_path = r"C:\Users\mehra\Downloads\Detection_of_Potholes\modelyolo8.pt"  
    data_yaml = r"C:\Users\mehra\Downloads\Detection_of_Potholes\Pothole_detection.v8i.yolov8\data.yaml"   
    
    try:
        metrics = evaluate_yolo_model(model_path, data_yaml)
        
        # Optional: Save detailed metrics to file
        with open("evaluation_report.txt", "w") as f:
            f.write(f"YOLO Model Evaluation Report\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            f.write(f"Precision: {metrics.box.p[0]:.4f}\n")
            f.write(f"Recall: {metrics.box.r[0]:.4f}\n")
        
        print("\n✅ Detailed report saved to 'evaluation_report.txt'")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Please check your model and data paths")