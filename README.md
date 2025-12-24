# Vietnamese License Plate Recognition

A system for detecting and recognizing Vietnamese license plates using YOLOv5 and YOLOv11. Supports both single-line and two-line plate formats.

## Installation

```bash
pip install -r requirement.txt
```

Pretrained models are included in the `model/` folder.

**For YOLOv5:** Download yolov5 (old version) from [this link]([https://drive.google.com/drive/folders/1W2NNPIjbt67RXvtUF2C3dYuqAAkogj4V?usp=sharing](https://drive.google.com/drive/folders/125DgE9RuFEF4tqkdR-WXzfmI45R_ZLEb?usp=sharing)) and extract it to the project root directory.

## Usage

### Image Processing
```bash
python lp_image_v11.py -i ./images/your_image.jpg
```

### Real-time Webcam/Video
```bash
python webcam_v11.py
```

Results are saved in `result_images/` and `result_video/` folders respectively.
