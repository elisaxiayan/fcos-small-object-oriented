import json
import os
import cv2
import numpy as np

def cv_imread(file_path):
    """支持中文路径的图片读取"""
    try:
        # 使用 numpy 读取字节流，再由 OpenCV 解码，避开路径编码问题
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图片出错: {file_path}, 错误: {e}")
        return None
def cv_imwrite(file_path, img):
    """支持中文路径的图片保存"""
    try:
        # 获取文件后缀
        ext = os.path.splitext(file_path)[1]
        # 将图片按格式编码后写入文件
        result, nparray = cv2.imencode(ext, img)
        if result:
            nparray.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"保存图片出错: {file_path}, 错误: {e}")
        return False

def visualize_scaled_annotations(json_dir, image_dir, output_dir, scale_factor=2):
    """
    读取JSON文件中的标注，将坐标乘以指定的缩放因子，并可视化。
    支持中文路径，支持 COCO 格式和自定义 Mask 格式。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"错误：在目录 {json_dir} 中未找到JSON文件。")
        return

    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        file_stem = os.path.splitext(json_file)[0]

        # 读取JSON文件（指定 utf-8 编码以支持中文内容）
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"错误：读取 {json_file} 时发生异常: {e}")
            continue

        # --- 1. 处理COCO格式的JSON (通常用于数据集) ---
        if 'images' in data and 'annotations' in data:
            images = {img['id']: img for img in data.get('images', [])}
            annotations_by_image = {img_id: [] for img_id in images.keys()}

            for anno in data.get('annotations', []):
                image_id = anno.get('image_id')
                if image_id in annotations_by_image:
                    annotations_by_image[image_id].append(anno)

            for image_id, image_info in images.items():
                file_name = image_info['file_name']
                image_path = os.path.join(image_dir, file_name)

                if not os.path.exists(image_path):
                    continue

                img = cv_imread(image_path)
                if img is None: continue

                # 获取当前图片的标注列表
                current_annos = annotations_by_image.get(image_id, [])
                for anno in current_annos:
                    bbox = anno['bbox']  # [x_min, y_min, width, height]
                    x_min, y_min, w, h = bbox

                    x1 = int(x_min * scale_factor)
                    y1 = int(y_min * scale_factor)
                    x2 = int((x_min + w) * scale_factor)
                    y2 = int((y_min + h) * scale_factor)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                output_path = os.path.join(output_dir, file_name)
                cv_imwrite(output_path, img)
                print(f"已处理(COCO): {file_name}")

        # --- 2. 处理单张图片自定义格式 (如 LabelMe 或特定格式) ---
        else:
            # 尝试获取图片文件名
            image_file = data.get('image', file_stem + '.png')
            image_file = os.path.basename(image_file)
            image_path = os.path.join(image_dir, image_file)

            # 自动纠正后缀名丢失或不匹配的问题
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpg', '.jpeg', '.BMP', '.PNG', '.JPG']:
                    temp_path = os.path.join(image_dir, file_stem + ext)
                    if os.path.exists(temp_path):
                        image_path = temp_path
                        image_file = file_stem + ext
                        break

            if not os.path.exists(image_path):
                print(f"警告：图片 {image_file} 不存在，跳过。")
                continue

            img = cv_imread(image_path)
            if img is None:
                continue

            # 处理 masks 字段
            if 'masks' in data:
                masks = data['masks']
                for mask in masks:
                    if not mask: continue

                    # 提取所有坐标并计算外接矩形
                    xs = [pt[0] for pt in mask]
                    ys = [pt[1] for pt in mask]

                    x1 = int(min(xs) * scale_factor)
                    y1 = int(min(ys) * scale_factor)
                    x2 = int(max(xs) * scale_factor)
                    y2 = int(max(ys) * scale_factor)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            output_path = os.path.join(output_dir, image_file)
            cv_imwrite(output_path, img)
            print(f"已处理并保存: {image_file}")

    print("\n✅ 所有图片处理完成！")


# --- 用户配置 ---
# 请确保路径前缀带有 r 以正确解析反斜杠
JSON_DIRECTORY = r"D:\BaiduNetdiskDownload\曹可馨\result\result-extra\labels\json"
IMAGE_DIRECTORY = r"D:\BaiduNetdiskDownload\曹可馨\result\result-extra\RGB"
OUTPUT_DIRECTORY = r"D:\BaiduNetdiskDownload\曹可馨\result\result-extra\vis-labeledRGB"

# 可视化缩放因子 (RGB通常设为2)
SCALE_FACTOR = 2

if __name__ == "__main__":
    visualize_scaled_annotations(JSON_DIRECTORY, IMAGE_DIRECTORY, OUTPUT_DIRECTORY, SCALE_FACTOR)