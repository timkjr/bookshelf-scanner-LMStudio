from ultralytics import YOLO, SAM
from utils import get_image_path

yolo_model = YOLO("models/yolo11x-seg.pt")
sam_model = SAM("models/mobile_sam.pt")

images = [
    get_image_path("img_1.jpg"),
    get_image_path("img_2.jpg"),
    get_image_path("img_3.jpg"),
    get_image_path("img_4.jpg"),
    get_image_path("img_5.jpg"),
]

#results = sam_model.predict(get_image_path("img_1.jpg"), points=[[400, 370], [900, 370]], labels=[1, 1])

yolo_results = yolo_model.predict(get_image_path("img_1.jpg"), imgsz=2560, conf=0.25)

for i, result in enumerate(yolo_results):
        result.show(f"Result {i + 1}")

# for box in yolo_results.xyxy[0]:  # Iterate over detected boxes
#     x1, y1, x2, y2 = map(int, box[:4])
#     box_center = [(x1 + x2) // 2, (y1 + y2) // 2]
#     sam_results = sam_model.predict(get_image_path("img_1.jpg"), points=[box_center], labels=[1])
#     for i, result in enumerate(sam_results):
#         result.show(f"Result {i + 1}")
