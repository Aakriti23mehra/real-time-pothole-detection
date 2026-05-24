import cv2
import torch
from ultralytics import YOLO
import PIL
import yaml
import tqdm

print(" All imports successful!")
print("OpenCV version:", cv2.__version__)
print("Torch version:", torch.__version__)
print("YOLO model loaded:", hasattr(YOLO, '__call__'))
