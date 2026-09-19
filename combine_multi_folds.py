import os
import shutil
from tqdm import tqdm  # 用于显示进度条，如果没有请运行 pip install tqdm


def merge_datasets_direct(source_root, target_root, move_files=False):
    """
    将 source_root 下的多个子文件夹直接合并到 target_root。
    假设文件名不冲突，直接复制/移动文件。

    :param move_files: 如果为 True，则剪切文件（节省空间）；如果为 False，则复制文件（保留原数据）。
    """

    # 1. 获取所有子文件夹
    subfolders = [f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))]

    # 排除输出目录本身（如果它在源目录内）
    target_abs = os.path.abspath(target_root)
    subfolders = [f for f in subfolders if os.path.abspath(os.path.join(source_root, f)) != target_abs]

    print(f"检测到 {len(subfolders)} 个子文件夹，开始{'移动' if move_files else '复制'}...")

    files_processed = 0

    # 2. 遍历每个子文件夹
    for sub_name in tqdm(subfolders, desc="合并进度"):
        sub_path = os.path.join(source_root, sub_name)

        # 3. 遍历子文件夹内部结构 (RGB, labels/json 等)
        for root, dirs, files in os.walk(sub_path):
            # 获取当前路径相对于子文件夹的相对结构
            # 比如: root 是 ".../Sub1/labels/json"，rel_path 就是 "labels/json"
            rel_path = os.path.relpath(root, sub_path)

            # 构建目标路径
            dest_dir = os.path.join(target_root, rel_path)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)

                # --- 安全检查 ---
                # 虽然你说没有冲突，但为了防止意外覆盖导致数据丢失，这里加一个简单的检查
                if os.path.exists(dst_file):
                    print(f"\n[警告] 文件冲突跳过: {dst_file} 已存在 (来自其他子文件夹)")
                    continue

                # 执行操作
                if move_files:
                    shutil.move(src_file, dst_file)
                else:
                    shutil.copy2(src_file, dst_file)  # copy2 保留文件创建时间等元数据

                files_processed += 1

    print(f"\n操作完成！共处理了 {files_processed} 个文件。")
    print(f"结果保存在: {target_root}")


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 输入路径
    input_path = r"F:\BaiduNetdiskDownload\1.23\第一组\1.23-赵晨雨\Result"

    # 输出路径
    output_path = r"F:\BaiduNetdiskDownload\1.23\第一组\zcy-test"

    # 是否移动文件？
    # True = 剪切（原文件消失，速度快，不占额外磁盘空间）
    # False = 复制（原文件保留，更安全）
    # MOVE_MODE = True
    MOVE_MODE = False

    # ===========================================

    if not os.path.exists(input_path):
        print("错误：输入路径不存在")
    else:
        merge_datasets_direct(input_path, output_path, move_files=MOVE_MODE)