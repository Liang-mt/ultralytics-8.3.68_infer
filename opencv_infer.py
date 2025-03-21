import argparse

import cv2
import numpy as np

# COCO数据集80类别名称（与YOLOv8输出顺序一致）
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, colors):
    """绘制单个检测框
    :param img: 输入图像
    :param class_id: 类别ID
    :param confidence: 置信度
    :param x: 左上角x坐标
    :param y: 左上角y坐标
    :param x_plus_w: 右下角x坐标
    :param y_plus_h: 右下角y坐标
    :param colors: 颜色列表
    """
    color = colors[class_id]
    label = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
    # 绘制边界框
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # 添加标签（调整位置防止超出图像边界）
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image, conf_thres=0.25, iou_thres=0.45):
    """主函数
    :param onnx_model: ONNX模型路径
    :param input_image: 输入图片路径
    :param conf_thres: 置信度阈值 (默认: 0.45)
    :param iou_thres: NMS IoU阈值 (默认: 0.60)
    """
    # 初始化模型
    model = cv2.dnn.readNetFromONNX(onnx_model)

    # 读取图像
    original_image = cv2.imread(input_image)
    if original_image is None:
        raise FileNotFoundError(f"图像文件未找到: {input_image}")

    # 创建正方形画布（保持长宽比）
    height, width = original_image.shape[:2]
    max_length = max(height, width)
    padded_image = np.zeros((max_length, max_length, 3), dtype=np.uint8)
    padded_image[0:height, 0:width] = original_image
    scale = max_length / 640  # 假设模型输入为640x640

    # 预处理（归一化 + 调整尺寸）
    blob = cv2.dnn.blobFromImage(
        padded_image,
        scalefactor=1 / 255.0,
        size=(640, 640),
        swapRB=True
    )
    model.setInput(blob)

    # 执行推理
    outputs = model.forward()

    # 解析输出
    outputs = np.array([cv2.transpose(outputs[0])])  # 调整维度
    rows = outputs.shape[1]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        max_score = np.max(classes_scores)

        # 使用传入的置信度阈值过滤
        if max_score >= conf_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[0][i][0], outputs[0][i][1], outputs[0][i][2], outputs[0][i][3]
            boxes.append([x - w / 2, y - h / 2, w, h])  # 转换为xywh格式
            scores.append(max_score)
            class_ids.append(class_id)

    # 应用NMS（使用传入的IoU阈值）
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres, 0.5)

    # 生成随机颜色
    colors = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

    # 绘制结果
    detections = []
    for i in indices:
        box = boxes[i]
        # 缩放回原始尺寸
        scaled_box = [round(v * scale) for v in box]
        x, y, w, h = scaled_box
        # 记录检测结果
        detections.append({
            "class_id": class_ids[i],
            "class_name": COCO_CLASSES[class_ids[i]],
            "confidence": scores[i],
            "box": scaled_box
        })
        # 绘制到原始图像
        draw_bounding_box(
            original_image, class_ids[i], scores[i],
            x, y, x + w, y + h, colors
        )

    # 显示结果
    cv2.imshow("Detection Results", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 目标检测推理脚本")
    parser.add_argument("--model",default="./model/yolo11n.onnx", help="ONNX模型路径 (默认: ./model/yolo11n.onnx)")
    parser.add_argument("--img", default="./images/bus.jpg", help="输入图片路径 (默认: ./images/bus.jpg)")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="置信度阈值 (范围: 0.0-1.0, 默认: 0.45)")
    parser.add_argument("--iou-thres", type=float, default=0.60, help="NMS IoU阈值 (范围: 0.0-1.0, 默认: 0.60)")
    args = parser.parse_args()


    main(args.model, args.img, args.conf_thres, args.iou_thres)