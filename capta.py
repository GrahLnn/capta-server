from super_gradients.training import models
import torch
import supervision as sv
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
import io
import base64

app = Flask(__name__)

# rf = Roboflow(api_key="AdwMbQISBDo540lfKcP1")
# project = rf.workspace("lingusic").project("capta")
# dataset = project.version(1).download("yolov5")


# LOCATION = dataset.location
# print("location:", LOCATION)
# print(onnxruntime.__version__)


def open_image(file):
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    elif isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    img = img.convert("RGB")
    return img


def detection(image):
    # image = open_image(image)
    CLASSES = sorted(["char", "target"])
    print("classes:", CLASSES)

    # 设置设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(DEVICE))

    # 加载模型
    MODEL_ARCH = "yolo_nas_l"
    best_model = models.get(
        MODEL_ARCH,
        num_classes=len(CLASSES),
        checkpoint_path="average_model.pth",
    ).to(DEVICE)

    image_array = np.asarray(bytearray(image), dtype=np.uint8)
    # image_array = np.array(image)

    # 使用 OpenCV 从 numpy 数组中解码图片
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # image = cv2.imread(image_path)
    CONFIDENCE_THRESHOLD = 0.5

    # 进行推理
    result = best_model.predict(image, conf=CONFIDENCE_THRESHOLD)

    # 提取检测结果
    detections = sv.Detections(
        xyxy=result.prediction.bboxes_xyxy,
        confidence=result.prediction.confidence,
        class_id=result.prediction.labels.astype(int),
    )

    detection_list = []

    for bbox, confidence, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        if confidence >= CONFIDENCE_THRESHOLD:
            # 转换坐标为整数
            x_min, y_min, x_max, y_max = map(int, bbox)

            # 类别名称
            class_name = CLASSES[class_id]

            # 创建字典并添加到列表
            detection_list.append(
                {
                    "crop": [x_min, y_min, x_max, y_max],
                    "classes": class_name,
                    "prob": float(confidence),
                }
            )
    # print(detection_list)
    # targets = [i.get("crop") for i in detection_list if i.get("classes") == "target"]
    # chars = [i.get("crop") for i in detection_list if i.get("classes") == "char"]
    # # 根据坐标进行排序
    # chars.sort(key=lambda x: x[0])

    # img = open_image(image_path)
    # chars = [img.crop(char) for char in chars]
    # folder_name = "chars"
    # os.makedirs(folder_name, exist_ok=True)
    # for idx, char in enumerate(chars):
    #     # 根据裁剪区域裁剪图像
    #     # cropped_image = img.crop(char)

    #     # 为裁剪后的图像生成一个文件名
    #     file_name = os.path.join(folder_name, f"char_{idx}.png")

    #     # 保存裁剪后的图像
    #     char.save(file_name)
    return detection_list


# pre = PreONNX("./modal/pre_model_v3.onnx", providers=["CPUExecutionProvider"])
# result = []
# for m, img_char in enumerate(chars):
#     if len(targets) == 0:
#         break
#     elif len(targets) == 1:
#         slys_index = 0
#     else:
#         img_target_list = []
#         for n, target in enumerate(targets):
#             img_target = img.crop(target)
#             img_target_list.append(img_target)
#         slys = pre.reason_all(img_char, img_target_list)
#         slys_index = slys.index(max(slys))
#     result.append(targets[slys_index])
#     targets.pop(slys_index)
#     if len(targets) == 0:
#         break

# print(result)

# labels = [
#     f"{CLASSES[class_id]} {confidence:.2f}"
#     for _, confidence, class_id in zip(
#         detections.xyxy, detections.confidence, detections.class_id
#     )
# ]
# # 绘制检测框
# box_annotator = sv.BoxAnnotator()
# # 绘制注释
# annotated_image = box_annotator.annotate(
#     scene=image.copy(), detections=detections, labels=labels, skip_label=False
# )

# # 显示注释后的图像
# # plt.imshow(annotated_image)
# # plt.axis("off")  # 关闭坐标轴
# # plt.show()

# # 保存注释后的图像
# output_image_path = "./annotated_image.png"
# cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# image_path = "./captcha.png"
# res = detection(image_path)
# print(res)


@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        image_base64 = data["image_base64"]
        image_byte = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_byte))
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_png_byte = output.getvalue()

        result = detection(image_png_byte)
        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
