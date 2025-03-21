# 导入库
from ultralytics import YOLO
import cv2

# 初始化检测器模型
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# 视频路径
video_path = "./data/input/traffic_signs.mp4"

# 读取视频文件
cap = cv2.VideoCapture(video_path)

# 检查错误
if not cap.isOpened():
    print("无法打开输入文件")
    exit()

# 处理
ret = True

while ret:
    ret, frame = cap.read()

    detections = detector(frame)

    for detection in detections:
        for bbox in detection.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]  # 获取边界框坐标

            # 转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制矩形
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("交通标志检测器", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# 取消注释以下行以检测并保存带注释的图像而不进行可视化
results = detector(video_path, save=True)