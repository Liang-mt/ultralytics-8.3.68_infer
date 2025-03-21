import argparse

import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8:
    """YOLOv8 目标检测模型类，处理推理和可视化"""

    # 硬编码的COCO数据集80个类别名称（按官方顺序）
    CLASSES = [
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

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化模型参数
        :param onnx_model: ONNX模型路径
        :param input_image: 输入图片路径
        :param confidence_thres: 置信度阈值
        :param iou_thres: NMS IoU阈值
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # 为每个类别生成随机颜色
        self.color_palette = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def draw_detections(self, img, box, score, class_id):
        """在图像上绘制检测框和标签"""
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        # 绘制矩形框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建标签文本
        label = f"{self.CLASSES[class_id]}: {score:.2f}"
        # 计算标签尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签位置
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # 绘制标签背景
        cv2.rectangle(
            img, (int(x1), int(label_y - label_height)),
            (int(x1 + label_width), int(label_y)), color, cv2.FILLED
        )
        # 添加标签文字
        cv2.putText(img, label, (int(x1), int(label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """图像预处理"""
        # 读取图像并检查有效性
        self.img = cv2.imread(self.input_image)
        if self.img is None:
            raise FileNotFoundError(f"图像文件未找到: {self.input_image}")

        # 获取原始尺寸
        self.img_height, self.img_width = self.img.shape[:2]

        # 转换颜色空间并调整尺寸
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))

        # 归一化并转换维度格式 (HWC -> CHW)
        image_data = np.array(img_resized) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def postprocess(self, input_image, output):
        """后处理：解析输出并应用NMS"""
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        # 计算尺寸缩放比例
        x_scale = self.img_width / self.input_width
        y_scale = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            # 过滤低置信度检测
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)

                # 还原原始坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_scale)
                top = int((y - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 绘制最终检测结果
        for i in indices:
            self.draw_detections(input_image, boxes[i], scores[i], class_ids[i])

        return input_image

    def main(self):
        """主推理流程"""
        # 创建ONNX Runtime会话
        session = ort.InferenceSession(
            self.onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # 获取输入尺寸
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 预处理 → 推理 → 后处理
        img_data = self.preprocess()
        outputs = session.run(None, {model_inputs[0].name: img_data})
        return self.postprocess(self.img.copy(), outputs)


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model/yolo11n.onnx", help="ONNX模型路径")
    parser.add_argument("--img", default="./images/bus.jpg", help="输入图片路径")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.60, help="NMS IoU阈值")
    args = parser.parse_args()

    # 初始化检测器
    detector = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)
    # 执行推理
    result_image = detector.main()

    # 显示结果
    cv2.imshow("Output", result_image)
    cv2.waitKey(0)
