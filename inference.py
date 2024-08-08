import onnxruntime as ort
import numpy as np
import cv2
import time

# 設定 YOLO 的輸入大小
yolo_width, yolo_height = 640, 640

# 圖像預處理
def preprocess(image, input_size=(yolo_width, yolo_height)):
    # 調整圖像大小並進行標準化
    resized = cv2.resize(image, input_size)
    resized = resized.astype(np.float32)
    resized /= 255.0
    resized = np.transpose(resized, (2, 0, 1))
    resized = np.expand_dims(resized, axis=0)
    return resized

# 解碼預測結果並進行 NMS
def decode_predictions(predictions, conf_threshold=0.5, iou_threshold=0.4):
    boxes = []
    scores = []
    class_ids = []

    # 預測結果的形狀為 (1, 1, 25200, 85)
    predictions = predictions[0][0]  # 去掉 batch_size 和單通道維度，變成 (25200, 85)
    
    # 解碼每個預測結果
    for det in predictions:
        # det 的形狀為 (85,)
        confidence = det[4]
        
        if confidence > conf_threshold:
            # 提取框框坐標和類別分數
            box = det[:4]  # (x_center, y_center, width, height)
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            score = confidence * class_scores[class_id]
            
            # 將邊界框坐標轉換為 (x_center, y_center, width, height)
            boxes.append(box)
            scores.append(float(score))
            class_ids.append(class_id)

    # 應用非最大抑制 (NMS)
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)

    final_boxes = []
    final_scores = []
    final_class_ids = []

    for i in indices:
        i = i[0]
        final_boxes.append(boxes[i])
        final_scores.append(scores[i])
        final_class_ids.append(class_ids[i])

    return final_boxes, final_scores, final_class_ids

# 繪製邊界框到圖像上
def draw_boxes(image, boxes, scores, class_ids, class_names):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x_center, y_center, width, height = box
        h, w, _ = image.shape
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # 確保坐標在圖像範圍內
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 繪製邊界框和標籤
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# 將YOLO邊界框從640x640轉換為原始圖片大小
def yolo_to_orig(box, orig_height, orig_width, yolo_width, yolo_height):
    x_center, y_center, width, height = box
    x_center = (x_center / yolo_width) * orig_width
    y_center = (y_center / yolo_height) * orig_height
    width = (width / yolo_width) * orig_width
    height = (height / yolo_height) * orig_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return (x1, y1, x2, y2)

def process(img_path, out_path):
    # 讀取和處理圖像
    image_path =  img_path # 替換為你的圖像路徑
    image = cv2.imread(image_path)
    input_tensor = preprocess(image)

    # 模型推理
    t1 = time.time()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})

    t2 = time.time()
    time_taken = t2 - t1
    print(f"CPU cost time {time_taken:.4f}")
    
    # 解碼和 NMS
    predictions = outputs  # 假設輸出形狀為 (1, 1, 25200, 85)
    boxes, scores, class_ids = decode_predictions(predictions)

    # 繪製邊界框
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = yolo_to_orig(box, image.shape[0], image.shape[1], yolo_width, yolo_height)
        class_id = class_ids[i]
        score = scores[i]
        label = f"{class_names[class_id]}: {score:.2f}"
        
        # 繪製矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 使用綠色的矩形框

        # 繪製分數和類別
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存圖片（可選）
    cv2.imwrite(out_path, image)
    print(f"Detection result saved to {out_path}")
    
    
if __name__ == "__main__":
    # coco 80類別
    cuda = False
    class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                    'hair drier', 'toothbrush']

    # 加載 YOLOv7 的 ONNX 模型
    model_path = './weights/yolov7.onnx'  # 替換為你的模型路徑
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    process('./inference/img/bus.jpg', 'output_bus.jpg')
    process('./inference/img/horses.jpg', 'output_horses.jpg')

    
