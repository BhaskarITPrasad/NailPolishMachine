from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import sys
import os

def process_image(image_path):
    # Convert RGBA to RGB if needed
    rgba_image = Image.open(image_path)
    rgb_image = rgba_image.convert("RGB")

    jpeg_image_path = "processed_image.jpg"
    rgb_image.save(jpeg_image_path, "JPEG")

    # Initialize Roboflow
    rf = Roboflow(api_key="CDgdYYw5br60mw9ZRBqG")
    project = rf.workspace().project("nail-gm5bx")
    model = project.version(1).model

    # Perform prediction
    result = model.predict(jpeg_image_path, confidence=40).json()

    # Save predictions to text file
    with open("predictions.txt", "w") as file:
        for item in result['predictions']:
            if 'points' in item:
                for point in item['points']:
                    file.write(f"{point['x']},{point['y']}\n")
    print("Coordinates saved to predictions.txt")

    # Annotate the image
    detections = sv.Detections.from_inference(result)
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    image = cv2.imread(jpeg_image_path)
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Save the annotated image to static folder for web display
    output_path = "static/annotated_output.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved at {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_image(image_path)
    else:
        print("No image path provided.")
