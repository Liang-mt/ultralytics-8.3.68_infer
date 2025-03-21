# 导入必要的库
import cv2
from ultralytics import YOLO

# 初始化检测器并加载权重
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# 图像路径
img_path = "./data/input/stop_sign.jpg"

# 可视化检测结果

# 读取图像
image = cv2.imread(img_path)

# detector.predict 返回一个检测对象的列表
detections = detector.predict(image)

for detection in detections:
    # 获取类别索引和类别名称
    class_ids = detection.boxes.cls  # cls 存储类别 ID

    # 遍历边界框
    for i, bbox in enumerate(detection.boxes):
        # 获取边界框坐标
        x1, y1, x2, y2 = bbox.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 根据类别 ID 获取类别名称
        class_id = int(class_ids[i])
        class_name = detection.names[class_id]  # detection.names 存储类别名称

        # 在图像上显示类别名称
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示带有检测结果的图像
        cv2.imshow("Detected", image)

        # 按 'q' 键退出
        if cv2.waitKey(10000) & 0xFF == ord("q"):
            break

# 取消注释以下行以检测并保存带注释的图像而不进行可视化
# results = detector(img_path, save=True)