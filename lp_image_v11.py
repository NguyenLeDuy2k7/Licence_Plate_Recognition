import cv2
import os
import argparse
from ultralytics import YOLO

import function.utils_rotate as utils_rotate
import function.helper_v11 as helper_v11


class Config:
    LP_DETECTOR_MODEL = 'model/LP_detector_v11.pt'
    LP_OCR_MODEL = 'model/LP_ocr_v11.pt'
    
    OUTPUT_DIR = "result_images"
    OUTPUT_PREFIX = "result_"
    
    DISPLAY_WIDTH = 700
    DISPLAY_HEIGHT = 700
    WINDOW_NAME = 'License Plate Recognition - YOLOv11'
    
    DETECTION_SIZE = 640
    LP_DETECTOR_CONFIDENCE = 0.50
    OCR_CONFIDENCE = 0.60


class VisualStyle:
    PLATE_BOX_COLOR = (0, 0, 225)
    TEXT_COLOR = (36, 255, 12)
    
    BOX_THICKNESS = 2
    TEXT_THICKNESS = 3
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    PLATE_TEXT_SCALE = 2
    FALLBACK_TEXT_POSITION = (7, 70)


def parse_arguments():
    parser = argparse.ArgumentParser(description='License Plate Recognition from Image')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    return parser.parse_args()


def initialize_models():
    detector = YOLO(Config.LP_DETECTOR_MODEL)
    detector.conf = Config.LP_DETECTOR_CONFIDENCE
    ocr_model = YOLO(Config.LP_OCR_MODEL)
    ocr_model.conf = Config.OCR_CONFIDENCE
    return detector, ocr_model


def setup_display_window():
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Config.WINDOW_NAME, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)


def detect_license_plates(detector, image):
    results = detector(image, imgsz=Config.DETECTION_SIZE)
    
    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = box.cls[0].item()
            plates.append([x1, y1, x2, y2, confidence, class_id])
    
    return plates


def process_plate(image, plate, ocr_model):
    x_min = int(plate[0])
    y_min = int(plate[1])
    width = int(plate[2] - plate[0])
    height = int(plate[3] - plate[1])
    
    crop_img = image[y_min:y_min + height, x_min:x_min + width]
    
    cv2.rectangle(
        image, 
        (int(plate[0]), int(plate[1])), 
        (int(plate[2]), int(plate[3])), 
        color=VisualStyle.PLATE_BOX_COLOR, 
        thickness=VisualStyle.BOX_THICKNESS
    )
    
    for contrast_mode in range(2):
        for center_threshold in range(2):
            deskewed = utils_rotate.deskew(crop_img, contrast_mode, center_threshold)
            plate_text = helper_v11.read_plate(ocr_model, deskewed)
            
            if plate_text != "unknown":
                text_position = (int(plate[0]), int(plate[1] - 10))
                cv2.putText(
                    image, 
                    plate_text, 
                    text_position, 
                    VisualStyle.FONT, 
                    VisualStyle.PLATE_TEXT_SCALE, 
                    VisualStyle.TEXT_COLOR, 
                    VisualStyle.TEXT_THICKNESS
                )
                return plate_text
    
    return None


def process_image_directly(image, ocr_model):
    plate_text = helper_v11.read_plate(ocr_model, image)
    
    if plate_text != "unknown":
        cv2.putText(
            image, 
            plate_text, 
            VisualStyle.FALLBACK_TEXT_POSITION, 
            VisualStyle.FONT, 
            VisualStyle.PLATE_TEXT_SCALE, 
            VisualStyle.TEXT_COLOR, 
            VisualStyle.TEXT_THICKNESS
        )
        return plate_text
    
    return None


def save_result(image, input_path: str) -> str:
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    input_filename = os.path.basename(input_path)
    output_path = os.path.join(Config.OUTPUT_DIR, f"{Config.OUTPUT_PREFIX}{input_filename}")
    
    cv2.imwrite(output_path, image)
    return output_path


def main():
    args = parse_arguments()
    
    detector, ocr_model = initialize_models()
    setup_display_window()
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Processing image: {args.image}")
    
    plates = detect_license_plates(detector, image)
    recognized_plates = set()
    
    if len(plates) == 0:
        plate_text = process_image_directly(image, ocr_model)
        if plate_text:
            recognized_plates.add(plate_text)
    else:
        for plate in plates:
            plate_text = process_plate(image, plate, ocr_model)
            if plate_text:
                recognized_plates.add(plate_text)
    
    print(f"Detected license plates: {recognized_plates}")
    
    output_path = save_result(image, args.image)
    print(f"Result saved to: {output_path}")
    
    cv2.imshow(Config.WINDOW_NAME, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
