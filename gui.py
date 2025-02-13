import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

# Load trained model configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only detecting roofs
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Path to trained weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if using GPU

# Create predictor
predictor = DefaultPredictor(cfg)

# Function to process image
def process_image(image_path):
    image = cv2.imread(image_path)
    outputs = predictor(image)

    # Visualize results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("roof_train"), scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert image format for display
    processed_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    return processed_image

# Function to open file dialog and process selected image
def upload_and_run():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")])
    if not file_path:
        return

    result_img = process_image(file_path)

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(result_img)
    plt.axis("off")
    plt.title("Roof Segmentation & Analysis")
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Solar Panel Suitability Analysis")

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Upload an aerial rooftop image:")
label.pack()

upload_button = tk.Button(frame, text="Upload & Analyze", command=upload_and_run)
upload_button.pack()

root.mainloop()
