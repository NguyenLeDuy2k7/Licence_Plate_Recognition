import math
from typing import Tuple, List, Any, Dict

POINT_TOLERANCE = 3
MIN_CHARACTERS = 7
MAX_CHARACTERS = 10

CHARACTER_CORRECTION_MAP: Dict[str, str] = {
    'T': '0', '0': '1', '1': '2', '2': '3', '3': '4',
    '4': '5', '5': '6', '6': '7', '7': '8', '8': '9',
    '9': 'A', 'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E',
    'E': 'F', 'F': 'G', 'G': 'H', 'H': 'K', 'I': 'L',
    'J': 'M', 'K': 'N', 'L': 'P', 'M': 'S', 'N': 'T',
    'O': 'U', 'P': 'V', 'Q': 'X', 'R': 'Y', 'Z': 'S'
}


def calculate_line_coefficients(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    y_intercept = y1 - (y2 - y1) * x1 / (x2 - x1)
    slope = (y1 - y_intercept) / x1
    return slope, y_intercept


def is_point_on_line(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    slope, y_intercept = calculate_line_coefficients(x1, y1, x2, y2)
    predicted_y = slope * x + y_intercept
    return math.isclose(predicted_y, y, abs_tol=POINT_TOLERANCE)


def _correct_character(class_name: str) -> str:
    return CHARACTER_CORRECTION_MAP.get(class_name, class_name)


def _extract_bounding_boxes_v11(results) -> List[List]:
    bounding_boxes = []
    
    for result in results:
        boxes = result.boxes
        class_names = result.names
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            raw_class_name = class_names[class_id]
            corrected_name = _correct_character(raw_class_name)
            
            bounding_boxes.append([x1, y1, x2, y2, confidence, class_id, corrected_name])
    
    return bounding_boxes


def _calculate_character_centers(bounding_boxes: List[List]) -> Tuple[List[List], float]:
    center_list = []
    y_sum = 0
    
    for box in bounding_boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        y_sum += y_center
        center_list.append([x_center, y_center, box[-1]])
    
    y_mean = int(y_sum / len(bounding_boxes))
    return center_list, y_mean


def _find_extreme_points(center_list: List[List]) -> Tuple[List, List]:
    leftmost = center_list[0]
    rightmost = center_list[0]
    
    for point in center_list:
        if point[0] < leftmost[0]:
            leftmost = point
        if point[0] > rightmost[0]:
            rightmost = point
    
    return leftmost, rightmost


def _determine_plate_type(center_list: List[List], leftmost: List, rightmost: List) -> str:
    plate_type = "1"
    
    if leftmost[0] == rightmost[0]:
        return plate_type
    
    for point in center_list:
        if not is_point_on_line(
            point[0], point[1],
            leftmost[0], leftmost[1],
            rightmost[0], rightmost[1]
        ):
            plate_type = "2"
            break
    
    return plate_type


def _format_single_line_plate(center_list: List[List]) -> str:
    sorted_chars = sorted(center_list, key=lambda x: x[0])
    return "".join(str(char[2]) for char in sorted_chars)


def _format_two_line_plate(center_list: List[List], y_mean: float) -> str:
    upper_line = []
    lower_line = []
    
    for point in center_list:
        if int(point[1]) > y_mean:
            lower_line.append(point)
        else:
            upper_line.append(point)
    
    upper_sorted = sorted(upper_line, key=lambda x: x[0])
    lower_sorted = sorted(lower_line, key=lambda x: x[0])
    
    upper_text = "".join(str(char[2]) for char in upper_sorted)
    lower_text = "".join(str(char[2]) for char in lower_sorted)
    
    return f"{upper_text}-{lower_text}"


def read_plate(yolo_license_plate: Any, image: Any) -> str:
    results = yolo_license_plate(image)
    bounding_boxes = _extract_bounding_boxes_v11(results)
    
    num_chars = len(bounding_boxes)
    if num_chars == 0 or num_chars < MIN_CHARACTERS or num_chars > MAX_CHARACTERS:
        return "unknown"
    
    center_list, y_mean = _calculate_character_centers(bounding_boxes)
    
    leftmost, rightmost = _find_extreme_points(center_list)
    plate_type = _determine_plate_type(center_list, leftmost, rightmost)
    
    if plate_type == "2":
        return _format_two_line_plate(center_list, y_mean)
    else:
        return _format_single_line_plate(center_list)
