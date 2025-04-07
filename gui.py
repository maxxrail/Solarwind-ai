# Updated GUI with full pipeline and exact obstacle detection logic from notebook
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt

# Detectron2 Setup
cfg = get_cfg()
cfg.merge_from_file(r"C:\Users\Platon\Documents\GitHub\Solarwind-ai\Solarwind-ai\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml") # chnage to your path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = r"C:\Users\Platon\Documents\GitHub\Solarwind-ai\Solarwind-ai\output_inria\model_final.pth" # change to your path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # Adjust threshold for better accuracy
cfg.TEST.DETECTIONS_PER_IMAGE = 200 # Change for more detections
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Constants
# Constraits of Test Images
physical_width_m = 986.58
physical_height_m = 708.28

# Panel area in m²
panel_area_m2 = 1.6

# Process Image with model and Save Combined Mask
def process_image(image_path):
    image = cv2.imread(image_path) # Read image
    outputs = predictor(image) # Run model on image
    masks = outputs["instances"].pred_masks.to("cpu").numpy() # Get masks
    combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8) # Initialize combined mask
    for mask in masks: ## Iterate through masks
        combined_mask[mask > 0.5] = 255
    cv2.imwrite("predicted_mask.png", combined_mask) # Save combined mask
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("roof_train"), scale=0.8) # Visualize
    out = v.draw_instance_predictions(outputs["instances"].to("cpu")) # Draw predictions
    processed_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB) # Convert to RGB
    return image, processed_image, "predicted_mask.png" # Return original, processed image and mask path

# Obstacle detection logic
def analyze_mask(image_path, mask_path):
    image = cv2.imread(image_path) # Read original image
    mask = cv2.imread(mask_path, 0) # Read mask
    final_mask = np.zeros_like(mask) # Initialize final mask
    house_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find house contours

    for house_contour in house_contours: # Iterate through house contours
        x, y, w, h = cv2.boundingRect(house_contour)
        largest_roof = image[y:y+h, x:x+w]
        roof_mask = mask[y:y+h, x:x+w]
        masked_roof = cv2.bitwise_and(largest_roof, largest_roof, mask=roof_mask)

        gray_masked_roof = cv2.cvtColor(masked_roof, cv2.COLOR_BGR2GRAY)
        blurred_roof = cv2.GaussianBlur(gray_masked_roof, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred_roof, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours_in_roof, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_contours_img = np.zeros_like(masked_roof)

        min_obstacle_area = 50
        max_obstacle_area = (w * h) * 0.3

        filtered_contours = []
        for cnt in contours_in_roof:
            area = cv2.contourArea(cnt)
            if min_obstacle_area < area < max_obstacle_area:
                filtered_contours.append(cnt)

        cv2.drawContours(filled_contours_img, filtered_contours, -1, (0, 255, 0), thickness=cv2.FILLED)

        filled_gray = cv2.cvtColor(filled_contours_img, cv2.COLOR_BGR2GRAY)
        _, obstacle_mask = cv2.threshold(filled_gray, 1, 255, cv2.THRESH_BINARY)
        obstacle_mask_inv = cv2.bitwise_not(obstacle_mask)
        processed_roof_mask = cv2.bitwise_and(roof_mask, obstacle_mask_inv)
        final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], processed_roof_mask)

    processed_white_pixels = cv2.countNonZero(final_mask)
    mask_height, mask_width = final_mask.shape
    area_per_pixel = (physical_width_m / mask_width) * (physical_height_m / mask_height)
    white_area_m2 = processed_white_pixels * area_per_pixel
    max_panels = int(white_area_m2 // panel_area_m2 * 0.9) # 90% of usable area is the max
    safe_panels = int(max_panels * 0.8) # 80% of usbale area
    recommended_panels = int(max_panels * 0.6) # 60% of usable area
    return final_mask, white_area_m2, safe_panels, recommended_panels

# Upload & Analyze Button Logic
def upload_and_run():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")])
    if not file_path:
        return
    original_img, infer_img, mask_path = process_image(file_path)
    final_mask, area, safe, rec = analyze_mask(file_path, mask_path)

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img[:, :, ::-1])
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(infer_img)
    plt.axis("off")
    plt.title("Inference Result")
    plt.subplot(1, 3, 3)
    plt.imshow(final_mask, cmap='gray')
    plt.axis("off")
    plt.title(f"Usable Area: {area:.1f} m²\nMax amount of Panels: {safe}\n Recommended number of Panels: {rec}")
    plt.tight_layout()
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Solarwind.AI - Roof Suitability Analyzer")
root.configure(bg="#d9d9d9")

# Center the window
window_width, window_height = 600, 550
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = screen_width // 2 - window_width // 2
center_y = screen_height // 2 - window_height // 2
root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

# --- LOGO FRAME ---
logo_frame = tk.Frame(root, bg="#d9d9d9")
logo_frame.pack(pady=(40, 20))

mcgill_img = Image.open(r"C:\Users\Platon\Documents\GitHub\Solarwind-ai\Solarwind-ai\McGill.png") # change to your path
mcgill_img = mcgill_img.resize((100, 60), Image.Resampling.LANCZOS)
mcgill_photo = ImageTk.PhotoImage(mcgill_img)

main_logo_img = Image.open(r"C:\Users\Platon\Documents\GitHub\Solarwind-ai\Solarwind-ai\Logo.png") # change to your path
main_logo_img = main_logo_img.resize((130, 130), Image.Resampling.LANCZOS)
main_logo_photo = ImageTk.PhotoImage(main_logo_img)

panel_img = Image.open(r"C:\Users\Platon\Documents\GitHub\Solarwind-ai\Solarwind-ai\Panel.png") # change to your path
panel_img = panel_img.resize((90, 70), Image.Resampling.LANCZOS)
panel_photo = ImageTk.PhotoImage(panel_img)

tk.Label(logo_frame, image=mcgill_photo, bg="#d9d9d9").grid(row=0, column=0, padx=20)
tk.Label(logo_frame, image=main_logo_photo, bg="#d9d9d9").grid(row=0, column=1, padx=20)
tk.Label(logo_frame, image=panel_photo, bg="#d9d9d9").grid(row=0, column=2, padx=20)

# --- Upload Prompt ---
label = tk.Label(root, text="Upload an aerial rooftop image",
                 font=("Helvetica", 14), bg="#d9d9d9", fg="#2e7d32")
label.pack(pady=(10, 10))

# --- Upload Button ---
upload_button = tk.Button(root, text="Upload & Analyze",
                          font=("Helvetica", 12, "bold"),
                          bg="#2e7d32", fg="white",
                          activebackground="#388e3c",
                          padx=20, pady=10,
                          command=upload_and_run)
upload_button.pack(pady=(0, 20))

# Start GUI
root.mainloop()
