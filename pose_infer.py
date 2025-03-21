import os
os.environ['YOLO_VERBOSE'] = str(False)
import argparse
import cv2
import warnings
from ultralytics import YOLO
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

# 17个关键点连接顺序   第五个点5是 (0,0)  不知道啥情况
#第5个点是右耳，在不同的情况下也可能是其他点，其置信度可能低于阈值，导致坐标为(0,0)
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
#             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
#             [1, 2], [1, 3], [2, 4]] #[4, 6],[5, 7]，这个点位可以放在最后，也可以不加，本代码暂时不加

# def infer_pose(image, re_list):
#     # 创建一个存储x和y坐标的列表
#     x_values = [0] * 17
#     y_values = [0] * 17
#
#     for re in re_list:
#         if hasattr(re, 'keypoints'):
#             keypoints = re.keypoints
#             data = keypoints.data
#             for key in data:
#                 for i, point in enumerate(key.tolist()):
#                     x_values[i] = point[0]
#                     y_values[i] = point[1]
#
#                 # 绘制坐标点
#                 for i in range(len(x_values)):
#                     cv2.circle(image, (int(x_values[i]), int(y_values[i])), 5, (0, 0, 255), -1)  # 绘制红色的圆点
#                     print(i,int(x_values[i]), int(y_values[i]))
#
#                 # 绘制连接线
#                 for connection in skeleton:
#                     start_point = (int(x_values[connection[0] - 1]), int(y_values[connection[0] - 1]))
#                     end_point = (int(x_values[connection[1] - 1]), int(y_values[connection[1] - 1]))
#                     cv2.line(image, start_point, end_point, (0, 255, 0), thickness=2)  # 绘制绿色的连线
#         else:
#             return image
#     return image

# 正确的 COCO 关键点连接顺序（0-based 索引）
skeleton = [
    [0, 1],    # 鼻子 → 左眼
    [0, 2],    # 鼻子 → 右眼
    [1, 3],    # 左眼 → 左耳
    [2, 4],    # 右眼 → 右耳
    [5, 6],    # 左肩 → 右肩
    [5, 7],    # 左肩 → 左肘
    [7, 9],    # 左肘 → 左手腕
    [6, 8],    # 右肩 → 右肘
    [8, 10],   # 右肘 → 右手腕
    [5, 11],   # 左肩 → 左髋
    [6, 12],   # 右肩 → 右髋
    [11, 12],  # 左髋 → 右髋
    [11, 13],  # 左髋 → 左膝盖
    [13, 15],  # 左膝盖 → 左脚踝
    [12, 14],  # 右髋 → 右膝盖
    [14, 16],  # 右膝盖 → 右脚踝
]

#存储没有被画的点的索引序号
# 全局列表，用于存储低置信度关键点的索引
low_conf_indices = []

def infer_pose(image, num_keypoints, re_list):
    """
    根据 YOLO 姿态估计结果在图像上绘制关键点和骨架连接。

    参数:
        image (numpy.ndarray): 输入图像。
        num_keypoints (int): 要处理的关键点数量。
        re_list (list): 包含关键点的 YOLO 结果列表。

    返回:
        numpy.ndarray: 绘制了关键点和骨架的图像。
    """
    global low_conf_indices
    #low_conf_indices.clear()

    for re in re_list:
        if not hasattr(re, 'keypoints'):
            continue  # 跳过没有关键点的结果

        keypoints = re.keypoints
        if keypoints is None:
            continue  # 跳过空的关键点

        data = keypoints.data.cpu().numpy()  # 转换为 NumPy 数组 [num_persons, num_keypoints, 3]
        if data.shape[0] == 0:
            continue  # 未检测到人体

        for person_idx in range(data.shape[0]):
            person_data = data[person_idx]  # [num_keypoints, 3]
            if person_data.shape[0] != num_keypoints:  # 关键点数量不合法
                continue

            person_low_conf = []
            # 绘制关键点
            for kp_idx in range(num_keypoints):
                x, y, conf = person_data[kp_idx]
                if conf > 0.5:  # 只绘制置信度 > 0.5 的关键点
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                else:
                    person_low_conf.append(kp_idx)  # 存储低置信度关键点
            low_conf_indices.append(person_low_conf)

            # 绘制骨架连接
            for connection in skeleton:
                start_idx, end_idx = connection
                if start_idx >= num_keypoints-1 or end_idx >= num_keypoints-1:
                    continue  # 跳过非法索引
                start_conf = person_data[start_idx][2]
                end_conf = person_data[end_idx][2]
                if start_conf > 0.5 and end_conf > 0.5:  # 只绘制置信度 > 0.5 的连接
                    x1, y1 = person_data[start_idx][0], person_data[start_idx][1]
                    x2, y2 = person_data[end_idx][0], person_data[end_idx][1]
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image

def get_class(bbox, cls, conf):
    """
    创建一个字典来存储边界框、类别和置信度信息。

    参数:
        bbox (tuple): 边界框坐标 (x_min, y_min, x_max, y_max)。
        cls (int): 类别 ID。
        conf (float): 置信度分数。

    返回:
        dict: 包含边界框、类别和置信度的字典。
    """
    return {"bbox": bbox,
            "cls": cls,
            "conf": conf}

def infer(re):
    """
    从 YOLO 结果中提取边界框、类别和置信度信息。

    参数:
        re (list): YOLO 结果列表。

    返回:
        list: 包含边界框、类别和置信度的字典列表。
    """
    dic = []
    if not re:  # 如果 re 为空，返回空列表
        return dic

    for result in re:
        boxes = result.boxes  # 用于边界框输出的 Boxes 对象
        for box in boxes:
            x_min, y_min, x_max, y_max, conf, cls = box.data[0].tolist()
            bbox = x_min, y_min, x_max, y_max
            result_dic = get_class(bbox, cls, conf)
            dic.append(result_dic)
    return dic

def draw_img(names, img, dic_list):
    """
    在图像上绘制边界框和类别标签。

    参数:
        names (list): 类别名称列表。
        img (numpy.ndarray): 输入图像。
        dic_list (list): 包含边界框、类别和置信度的字典列表。

    返回:
        numpy.ndarray: 绘制了边界框和标签的图像。
    """
    if not dic_list:
        return img
    for result in dic_list:
        x_min, y_min, x_max, y_max = result["bbox"]
        cls = int(result["cls"])
        conf = result["conf"]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        text = f"{names[cls]} {conf:.2f}"
        cv2.putText(img, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    return img


def main(args):
    """
    主函数，根据提供的参数处理推理任务。

    参数:
        args (argparse.Namespace): 解析后的命令行参数。
    """
    weights = args.weights_path
    model = YOLO(weights)
    names = model.names
    num_keypoints = args.num_keypoints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从参数中获取推理参数
    predict_args = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": device
    }

    if args.demo == 'image':
        img = cv2.imread(args.image_path)
        results = model.predict(source=img, **predict_args)

        # 根据参数决定是否调用 infer 和 infer_pose
        if args.use_infer:
            dic_list = infer(results)
            img = draw_img(names, img, dic_list)
        if args.use_infer_pose:
            img = infer_pose(img, num_keypoints, results)
        if args.save:
            output_name = "result_" + os.path.basename(args.image_path)
            cv2.imwrite(output_name, img)
        cv2.imshow("Result", img)
        cv2.waitKey(0)


    elif args.demo == 'video':

        cap = cv2.VideoCapture(args.video_path)

        # 初始化 VideoWriter（仅在保存时生效）
        out = None

        if args.save:

            # 获取原视频参数
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(fps)
            # 处理无效的帧率（如摄像头返回0）
            if fps <= 0:
                fps = 30.0  # 默认帧率

            # 定义编码器并创建 VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或根据系统支持选择其他编码器
            output_path = ''
            if args.video_path in [0, 1]:
                output_path = "result.mp4"
            else:
                output_path = "result_" + os.path.basename(args.video_path)

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 摄像头镜像翻转
            if args.video_path in [0, 1]:
                frame = cv2.flip(frame, 1)
            # 模型推理
            results = model.predict(source=frame, **predict_args)
            # 后处理
            if args.use_infer:
                dic_list = infer(results)
                frame = draw_img(names, frame, dic_list)
            if args.use_infer_pose:
                frame = infer_pose(frame, num_keypoints, results)
            # 显示实时画面
            cv2.imshow('Video', frame)
            # 保存视频
            if args.save and out is not None:
                out.write(frame)
            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        if args.save and out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser(description='用于姿态估计的模型推理。')
    parser.add_argument('--demo', type=str, default='video', choices=['image', 'video'],help='推理类型：image 或 video。')
    parser.add_argument('--weights_path', type=str, default='./model/yolo11n-pose.pt', help='模型权重路径。')
    parser.add_argument('--image_path', type=str, default='./data/input/stop_sign.jpg', help='图像文件路径。')
    parser.add_argument('--video_path', type=str, default=0, help='视频文件路径或摄像头ID（0表示默认摄像头）。')
    parser.add_argument('--num_keypoints', type=int, default=17, help='关键点数量。')
    parser.add_argument('--save', type=int, default=True, help='是否保存。')
    parser.add_argument('--imgsz', type=int, default=640, help='推理尺寸（像素），默认640。')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值（0-1），默认0.6。')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值（0-1），默认0.45。')
    parser.add_argument('--use_infer', type=int, default=True, help='是否调用 infer 函数（绘制边界框和类别），默认True。')
    parser.add_argument('--use_infer_pose', type=int, default=True, help='是否调用 infer_pose 函数（绘制关键点和骨架），默认False。')

    args = parser.parse_args()
    main(args)
    #print(low_conf_indices)