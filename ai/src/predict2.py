import os
import numpy as np
from torch import Tensor
from ultralytics import YOLO, SAM
from utils import get_image_path, scale_image, rotate_if_spine, image_to_base64
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler


def main():
    #yolo_model = YOLO("models/yolo11x-seg.pt", task="segment")

    images = os.listdir("C:\\Users\\suxro\\source\\repos\\bookshelf-scanner\\ai\\output")
    outputs: list[str] = []

    chat_handler = MoondreamChatHandler.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="moondream2-mmproj-f16.gguf",
        verbose=False,
    )

    llm = Llama.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="moondream2-text-model-f16.gguf",
        chat_handler=chat_handler,
        n_ctx=2048,
    )

    for image in images:
        if not image.endswith(".png"):
            continue

        image_path = os.path.abspath(f"C:\\Users\\suxro\\source\\repos\\bookshelf-scanner\\ai\\output\\{image}")

        response = llm.create_chat_completion(messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Recognize the title and author of this book in the format 'Title by Author'. If there is no author, just the title is fine."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(image_path) } 
                }
            ]
        }])

        if not response:
            continue

        message_content: str = response["choices"][0]["message"]["content"] # type: ignore
        book_index = int(image.split("_")[1].split(".")[0])
        print(f"Book {book_index}:", message_content.strip())

    return

    original_image = Image.open(get_image_path("img_1.jpg"))
    scaled_image = scale_image(original_image, (2560, 2560))

    yolo_results = yolo_model.predict(scaled_image, imgsz=2560, half=True, classes=[73], retina_masks=True, conf=0.35)

    results = yolo_results[0]
    masks = results.masks  # Segmentation masks
    boxes = results.boxes  # Bounding boxes

    if masks is None or boxes is None:
        print("No books detected.")
        exit()

    results.show()
    outputs: list[str] = []

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

        output_image_path = os.path.abspath(f"output/book_{i + 1}.png")

        # Save the cropped image and mask (optional)
        cropped_image.save(output_image_path)

        response = ollama.chat(model="moondream", messages=[{
            "role": "user",
            "content": "Recognize the title and author of this book in the format 'Title by Author'. If there is no author, just the title is fine.",
            "images": [output_image_path]
        }])
        
        # Display or print the result
        print(f"Book {i + 1} Title:", output_image_path)
        outputs.append(f"Book {i + 1} Title: {response}")

    # Save the outputs to a text file
    with open("output/outputs.txt", "w") as f:
        f.write("\n".join(outputs))

if __name__ == "__main__":
    main()
