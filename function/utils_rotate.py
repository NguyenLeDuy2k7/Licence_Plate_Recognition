import numpy as np
import math
import cv2
from typing import Optional

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
CANNY_THRESHOLD_LOW = 30
CANNY_THRESHOLD_HIGH = 100
HOUGH_THRESHOLD = 30
MAX_ROTATION_ANGLE = 30
MIN_LINE_Y_POSITION = 7


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced_l = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, 
        rotation_matrix, 
        image.shape[1::-1], 
        flags=cv2.INTER_LINEAR
    )
    return rotated_image


def _get_image_dimensions(image: np.ndarray) -> tuple:
    if len(image.shape) == 3:
        height, width, _ = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
    else:
        raise ValueError("Unsupported image format")
    return height, width


def _find_dominant_line(lines: np.ndarray, height: int, center_threshold: int) -> int:
    min_line_y = 100
    min_line_index = 0
    
    for i, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            center_y = (y1 + y2) / 2
            
            if center_threshold == 1 and center_y < MIN_LINE_Y_POSITION:
                continue
            
            if center_y < min_line_y:
                min_line_y = center_y
                min_line_index = i
    
    return min_line_index


def compute_skew_angle(source_image: np.ndarray, center_threshold: int) -> float:
    height, width = _get_image_dimensions(source_image)
    
    blurred = cv2.medianBlur(source_image, 3)
    edges = cv2.Canny(
        blurred, 
        threshold1=CANNY_THRESHOLD_LOW, 
        threshold2=CANNY_THRESHOLD_HIGH, 
        apertureSize=3, 
        L2gradient=True
    )
    
    min_line_length = width / 1.5
    max_line_gap = height / 3.0
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=math.pi / 180, 
        threshold=HOUGH_THRESHOLD, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return 1.0
    
    dominant_line_index = _find_dominant_line(lines, height, center_threshold)
    
    angle_sum = 0.0
    valid_count = 0
    
    for x1, y1, x2, y2 in lines[dominant_line_index]:
        line_angle = np.arctan2(y2 - y1, x2 - x1)
        
        if math.fabs(line_angle) <= MAX_ROTATION_ANGLE:
            angle_sum += line_angle
            valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    average_angle = (angle_sum / valid_count) * 180 / math.pi
    return average_angle


def deskew(source_image: np.ndarray, apply_contrast: int, center_threshold: int) -> np.ndarray:
    if apply_contrast == 1:
        processed_image = enhance_contrast(source_image)
        skew_angle = compute_skew_angle(processed_image, center_threshold)
    else:
        skew_angle = compute_skew_angle(source_image, center_threshold)
    
    return rotate_image(source_image, skew_angle)


