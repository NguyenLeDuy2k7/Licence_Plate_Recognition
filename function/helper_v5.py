import math
from typing import Tuple, List, Any

POINT_TOLERANCE = 3
MIN_CHARACTERS = 7
MAX_CHARACTERS = 10


def calculate_line_coefficients(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    y_intercept = y1 - (y2 - y1) * x1 / (x2 - x1)
    slope = (y1 - y_intercept) / x1
    return slope, y_intercept


def is_point_on_line(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    slope, y_intercept = calculate_line_coefficients(x1, y1, x2, y2)
    predicted_y = slope * x + y_intercept
    return math.isclose(predicted_y, y, abs_tol=POINT_TOLERANCE)


def _extract_bounding_boxes(results) -> List[List]:
    return results.pandas().xyxy[0].values.tolist()


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
    bounding_boxes = _extract_bounding_boxes(results)
    
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