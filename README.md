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

Running the script will generate two annotated output images and display the inference time on the CPU.
```bash
python inference.py
```

## Result

```bash
CPU cost time 0.3624
Detection result saved to output_bus.jpg
CPU cost time 0.3358
Detection result saved to output_horses.jpg
```

<img src="inference/result/output_bus.jpg" alt="示例圖片" width="512" />

<img src="inference/result/output_horses.jpg" alt="示例圖片" width="512" />

## DEMO Video

[Watch the Demo video](https://youtu.be/zynkqfNjtjs)

