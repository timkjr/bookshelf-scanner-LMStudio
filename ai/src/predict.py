import re
import cv2
import numpy as np
import easyocr
from torch import Tensor
from ultralytics import YOLO
from utils import get_image_path, scale_image, rotate_if_spine
from PIL import Image, ImageEnhance

def main():
    # Initialize the YOLO model and EasyOCR reader
    yolo_model = YOLO("models/yolo11x-seg.pt", task="segment")
    reader = easyocr.Reader(["en"])

    # Load and preprocess the image
    image_path = get_image_path("img_1.jpg")
    scaled_image = load_and_scale_image(image_path)
    scaled_image = preprocess_image(scaled_image)
    scaled_image = reduce_noise(scaled_image)

    # Detect books in the image
    results = detect_books(scaled_image, yolo_model)
    results.show()

    # Process each detected book for OCR
    masks = results.masks
    boxes = results.boxes

    if masks is None or boxes is None:
        print("No books detected.")
        return

    # Loop over each detected book
    for i, (mask_data, box) in enumerate(zip(masks.data, boxes)):  # type: ignore
        mask_data: Tensor

        # Mask and crop the book from the original image
        cropped_image, bbox = mask_and_crop_book(scaled_image, mask_data, box)

        # Rotate if it's a spine
        cropped_image = rotate_if_spine(cropped_image, bbox)

        # Enhance the cropped image for OCR
        #enhanced_image = enhance_image(cropped_image)

        # Apply OCR
        book_title = apply_ocr(cropped_image, reader)
        book_title = clean_text(book_title)

        # Print and save the results
        print(f"Book {i + 1} Title: {book_title}")

        cropped_image.save(f"output/book_{i + 1}.png")
        #enhanced_image.save(f"output/book_{i + 1}_enhanced.png")

def load_and_scale_image(image_path: str, size = (2560, 2560)) -> Image.Image:
    """Load and scale the image to a specific size."""
    original_image = Image.open(image_path)
    scaled_image = scale_image(original_image, size)
    return scaled_image

def detect_books(image: Image.Image, model: YOLO):
    """Run YOLO model to detect books in the image and return results."""
    results = model.predict(image, imgsz=2560, half=True, classes=[73], retina_masks=True, conf=0.35)
    return results[0]

def mask_and_crop_book(image: Image.Image, mask_data: Tensor, box) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Apply mask to isolate the book and crop it using bounding box coordinates."""
    mask = Image.fromarray(mask_data.cpu().numpy().astype("uint8") * 255)
    masked_image = Image.new("RGB", image.size)
    masked_image.paste(image, mask=mask)

    # Crop to bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_image = masked_image.crop((x1, y1, x2, y2))
    return cropped_image, (x1, y1, x2, y2)

def enhance_image(image: Image.Image) -> Image.Image:
    """Enhance contrast and sharpness of the image for better OCR results."""
    image = ImageEnhance.Contrast(image).enhance(2)
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(2)
    #image = adaptive_threshold(image)
    return image

def apply_ocr(image: Image.Image, reader: easyocr.Reader) -> str:
    """Use EasyOCR to extract text from the image."""
    result = reader.readtext(np.array(image))

    filtered_results = [res for res in result]
    book_title = " ".join([res[1] for res in filtered_results])
    return book_title.strip()

def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance the image before detection."""
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.1)
    return image

def reduce_noise(image: Image.Image) -> Image.Image:
    """Apply noise reduction to the image."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

def adaptive_threshold(image: Image.Image) -> Image.Image:
    """Apply adaptive thresholding to the image."""
     # Convert to grayscale if not already
    image_cv = np.array(image)

    if len(image_cv.shape) == 3:  # Check if image has 3 channels (RGB)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    thresh_cv = cv2.adaptiveThreshold(
        image_cv, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    return Image.fromarray(thresh_cv)

def clean_text(text: str) -> str:
    """Clean the OCR output text."""
    text = re.sub(r"[^A-Za-z0-9\s\-\:\'\",\.]", "", text)
    return text.strip()

if __name__ == "__main__":
    main()
