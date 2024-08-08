# YOLOv7 ONNX

This project implements an object detection system using ONNX Runtime with a YOLOv7 pre-trained model. It requires a separate implementation of NMS (Non-Maximum Suppression) as it is not included within the model.

<details><summary><b>Related GitHub</b></summary>

* [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

</details>

## Installation Requirements

1. Clone the repository:

```bash
git clone https://github.com/Ghostly0328/yolov7_onnx.git
cd yolov7_onnx
mkdir weights
```

2. Download the [weights](https://drive.google.com/file/d/1EMm-pDLzujFfwFt2Bgj2t7WmgzxF2-wj/view?usp=sharing) and place them in the `./weights/` directory.

3. Build the Docker image:

```bash
docker build -t yolov7_onnx .
```

4. Run the Docker container:

```bash
docker run -it -v your_code_path/:/base yolov7_onnx bash
```

## Inference
```bash
python inference.py
```
