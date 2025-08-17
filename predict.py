import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import logging


def setup_logging():
    """Configure logging for prediction process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def predict_instance_segmentation(image_path, model_path):
    """Perform instance segmentation on the input image."""
    logger = setup_logging()

    try:
        # Validate input paths
        if not os.path.exists(image_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Invalid image or model path")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Inference on {device}")

        # Load trained YOLOv8 segmentation model
        model = YOLO(model_path)

        # Class colors for visualization
        colors = {
            "Streak": (0, 100, 255),  # Red
            "Spot": (255, 100, 10),  # Green
        }

        # Read input image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_height, original_width = img.shape[:2]

        # Perform inference
        logger.info(f"Running inference on: {image_path}")
        results = model(image_path)

        # Process segmentation results
        if results[0].masks is not None:
            logger.info(f"Masks detected: {len(results[0].masks.data)}")

            # Iterate through detected instances
            for i, mask in enumerate(results[0].masks.data):
                class_id = results[0].boxes.cls[i].item()
                class_name = model.names[int(class_id)]
                confidence = results[0].boxes.conf[i].item()

                # Convert mask to binary mask and resize to match input image dimensions
                mask_np = mask.cpu().numpy()
                binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                binary_mask = cv2.resize(
                    binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
                )

                # Find contours of the mask
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Get color for this class
                color = colors.get(class_name, (255, 255, 255))

                # Draw filled semi-transparent mask
                mask_overlay = img.copy()
                cv2.drawContours(mask_overlay, contours, -1, color, -1)
                cv2.addWeighted(mask_overlay, 0.3, img, 0.7, 0, img)

                # Draw contour outline
                cv2.drawContours(img, contours, -1, color, 2)

                # Get the centroid of the largest contour for placing the label
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        text_x = int(M["m10"] / M["m00"])
                        text_y = int(M["m01"] / M["m00"])
                    else:
                        text_x, text_y = largest_contour[0][0][0], largest_contour[0][0][1]

                    # Add label with class and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )

                    # Draw label background
                    cv2.rectangle(
                        img,
                        (text_x, text_y - label_height - 5),
                        (text_x + label_width, text_y),
                        color,
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        img,
                        label,
                        (text_x, text_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                    logger.info(f"Detected: {label}")

        # Save annotated image
        os.makedirs("predictions", exist_ok=True)
        output_path = os.path.join(
            "predictions", f"segmented_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, img)
        logger.info(f"Segmented image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def main():
    # Paths for model and image
    model_path = "runs/segmentation_training/LPBF_defect_segmentation/weights/best.pt"
    image_path = "images/train/1065138_A0020b.png"

    try:
        predict_instance_segmentation(image_path, model_path)
    except Exception as e:
        print(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
