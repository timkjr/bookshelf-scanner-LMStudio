import numpy as np
import pytesseract
import easyocr
from torch import Tensor
from ultralytics import YOLO, SAM
from utils import get_image_path, scale_image, rotate_if_spine
from PIL import Image, ImageEnhance

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

yolo_model = YOLO("models/yolo11x-seg.pt", task="segment")
#sam_model = SAM("models/mobile_sam.pt")
reader = easyocr.Reader(["en"])

original_image = Image.open(get_image_path("img_1.jpg"))
scaled_image = scale_image(original_image, (2560, 2560))

yolo_results = yolo_model.predict(scaled_image, imgsz=2560, half=True, classes=[73], retina_masks=True)

results = yolo_results[0]
masks = results.masks  # Segmentation masks
boxes = results.boxes  # Bounding boxes

if masks is None or boxes is None:
    print("No books detected.")
    exit()

results.show()

# Loop over each detected book
for i, (mask_data, box) in enumerate(zip(masks.data, boxes)): # type: ignore
    mask_data: Tensor

    # Convert mask to a PIL Image
    mask = Image.fromarray(mask_data.cpu().numpy().astype("uint8") * 255)

    # Create a new image for the masked book
    masked_image = Image.new("RGB", scaled_image.size)
    masked_image.paste(scaled_image, mask=mask)

    # Crop the image to the bounding box for efficiency
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_image = masked_image.crop((x1, y1, x2, y2))

    # Rotate the image if it's identified as a spine
    cropped_image = rotate_if_spine(cropped_image, (x1, y1, x2, y2))

    # Enhance contrast and sharpness
    #enhancer = ImageEnhance.Contrast(cropped_image)
    #cropped_image_enhanced = enhancer.enhance(2)

    #sharpness_enhancer = ImageEnhance.Sharpness(cropped_image_enhanced)
    #cropped_image_enhanced = sharpness_enhancer.enhance(2)

    # Apply OCR to extract text (book title)
    #book_title: str = pytesseract.image_to_string(cropped_image, lang="eng", config="--psm 7 --oem 3")

    result = reader.readtext(np.array(cropped_image))
    book_title = " ".join([res[1] for res in result])

    # Display or print the result
    print(f"Book {i + 1} Title: {book_title.strip()}")

    # Save the cropped image and mask (optional)
    cropped_image.save(f"output/book_{i + 1}.png")
