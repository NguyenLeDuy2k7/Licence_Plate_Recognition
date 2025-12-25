import cv2
import os
import time
from pathlib import Path
import torch

import function.utils_rotate as utils_rotate
import function.helper_v5 as helper_v5


class Config:
    YOLO_PATH = 'yolov5'
    LP_DETECTOR_MODEL = 'model/LP_detector_nano_v5.pt'
    LP_OCR_MODEL = 'model/LP_ocr_nano_v5.pt'
    
    VIDEO_SOURCE = "images\\video_2.mp4"
    
    OUTPUT_DIR = "result_video"
    OUTPUT_FILENAME = "output.mp4"
    
    DISPLAY_WIDTH = 500
    DISPLAY_HEIGHT = 500
    WINDOW_NAME = 'License Plate Recognition - YOLOv5'
    
    DETECTION_SIZE = 640
    LP_DETECTOR_CONFIDENCE = 0.50
    OCR_CONFIDENCE = 0.60
    DEFAULT_FPS = 30


class VisualStyle:
    PLATE_BOX_COLOR = (0, 0, 225)
    TEXT_COLOR = (36, 255, 12)
    FPS_COLOR = (100, 255, 0)
    
    BOX_THICKNESS = 2
    TEXT_THICKNESS = 2
    FPS_THICKNESS = 3
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    PLATE_TEXT_SCALE = 0.9
    FPS_TEXT_SCALE = 3
    
    FPS_POSITION = (7, 70)


def initialize_models():
    detector = torch.hub.load(
        Config.YOLO_PATH, 
        'custom', 
        path=Config.LP_DETECTOR_MODEL, 
        force_reload=True, 
        source='local'
    )
    detector.conf = Config.LP_DETECTOR_CONFIDENCE
    
    ocr_model = torch.hub.load(
        Config.YOLO_PATH, 
        'custom', 
        path=Config.LP_OCR_MODEL, 
        force_reload=True, 
        source='local'
    )
    ocr_model.conf = Config.OCR_CONFIDENCE
    
    return detector, ocr_model


def initialize_video_capture():
    capture = cv2.VideoCapture(Config.VIDEO_SOURCE)
    
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = Config.DEFAULT_FPS
    
    return capture, frame_width, frame_height, fps


def initialize_video_writer(frame_width: int, frame_height: int, fps: int):
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    input_path = Path(Config.VIDEO_SOURCE)
    if input_path.stem.isdigit():
        output_filename = Config.OUTPUT_FILENAME
    else:
        output_filename = f"{input_path.stem}_result{input_path.suffix}"
    
    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return writer, output_path


def setup_display_window():
    cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Config.WINDOW_NAME, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)


def detect_license_plates(detector, frame):
    results = detector(frame, size=Config.DETECTION_SIZE)
    return results.pandas().xyxy[0].values.tolist()


def process_plate(frame, plate, ocr_model):
    x_min = int(plate[0])
    y_min = int(plate[1])
    width = int(plate[2] - plate[0])
    height = int(plate[3] - plate[1])

    crop_img = frame[y_min:y_min + height, x_min:x_min + width]

    cv2.rectangle(
        frame, 
        (int(plate[0]), int(plate[1])), 
        (int(plate[2]), int(plate[3])), 
        color=VisualStyle.PLATE_BOX_COLOR, 
        thickness=VisualStyle.BOX_THICKNESS
    )

    for contrast_mode in range(2):
        for center_threshold in range(2):
            deskewed = utils_rotate.deskew(crop_img, contrast_mode, center_threshold)
            plate_text = helper_v5.read_plate(ocr_model, deskewed)
            
            if plate_text != "unknown":
                text_position = (int(plate[0]), int(plate[1] - 10))
                cv2.putText(
                    frame, 
                    plate_text, 
                    text_position, 
                    VisualStyle.FONT, 
                    VisualStyle.PLATE_TEXT_SCALE, 
                    VisualStyle.TEXT_COLOR, 
                    VisualStyle.TEXT_THICKNESS
                )
                return plate_text
    
    return None


def draw_fps(frame, fps: int):
    cv2.putText(
        frame, 
        str(fps), 
        VisualStyle.FPS_POSITION, 
        VisualStyle.FONT, 
        VisualStyle.FPS_TEXT_SCALE, 
        VisualStyle.FPS_COLOR, 
        VisualStyle.FPS_THICKNESS, 
        cv2.LINE_AA
    )


def main():
    detector, ocr_model = initialize_models()
    capture, frame_width, frame_height, fps = initialize_video_capture()
    writer, output_path = initialize_video_writer(frame_width, frame_height, fps)
    setup_display_window()
    
    prev_frame_time = 0
    
    print(f"Processing video: {Config.VIDEO_SOURCE}")
    print(f"Output will be saved to: {output_path}")
    print("Press 'q' to quit...")
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            plates = detect_license_plates(detector, frame)
            
            recognized_plates = set()
            for plate in plates:
                plate_text = process_plate(frame, plate, ocr_model)
                if plate_text:
                    recognized_plates.add(plate_text)
            
            current_time = time.time()
            if prev_frame_time > 0:
                fps_display = int(1 / (current_time - prev_frame_time))
                draw_fps(frame, fps_display)
            prev_frame_time = current_time
            
            writer.write(frame)
            cv2.imshow(Config.WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        writer.release()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
