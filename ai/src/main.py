from ultralytics import YOLO
from utils import get_image_path

model = YOLO("models/yolo11x.pt")

images = [
    get_image_path("img_1.jpg"),
    get_image_path("img_2.jpg"),
    get_image_path("img_3.jpg"),
    get_image_path("img_4.jpg"),
    get_image_path("img_5.jpg"),
]

results: list[YOLO] = model(images)

for i, result in enumerate(results):
    result.show(f"Result {i + 1}")
