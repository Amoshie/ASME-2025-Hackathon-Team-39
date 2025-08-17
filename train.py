import os
import sys
from ultralytics import YOLO
import torch
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def train_yolov8_segmentation(dataset_yaml):
    logger = setup_logging()
    
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        # Load a pretrained YOLOv8 segmentation model
        model = YOLO('yolov8n-seg.pt')
        
        # Advanced training parameters
        training_params = {
            'data': dataset_yaml,
            'epochs': 80,
            'batch': 16,
            'imgsz': 640,
            'device': device,
            'project': 'runs/segmentation_training',
            'name': 'LPBF_defect_segmentation',
            #'patience': 10,  # Early stopping
            'save_period': 1,  # Save checkpoint after every epoch
            'plots': False,
            'close_mosaic': 0,
            'exist_ok': True  # Overwrite existing results folder
        }
        
        # Start training
        logger.info("Starting YOLOv8 Instance Segmentation Training...")
        results = model.train(**training_params)
        
        logger.info("Training completed successfully.")
        logger.info(f"Best model saved at: {results.best}")
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

def main():
    dataset_yaml = 'dataset.yaml'
    
    try:
        train_yolov8_segmentation(dataset_yaml)
    except Exception as e:
        print(f"Training process terminated with an error: {e}")

if __name__ == "__main__":
    main()