import os
import shutil
import argparse
import json
from pathlib import Path

# 默认目录设置（用户可以直接修改这里的默认值）
DEFAULT_SOURCE_DIR = r"D:\BaiduNetdiskDownload\宁伟博(22)\result"
DEFAULT_TARGET_DIR = r"D:\BaiduNetdiskDownload\宁伟博(22)\result-extra"


class FileProcessor:
    def __init__(self, source_dir, target_dir, overwrite=False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.overwrite = overwrite
        self.required_folders = ['HSI', 'RGB', 'IR', 'labels']
        self.common_files = set()
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0

    def validate_source_directory(self):
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.source_dir}")

        for folder in self.required_folders:
            folder_path = self.source_dir / folder
            if not folder_path.exists():
                raise FileNotFoundError(f"源目录中缺少必要的子文件夹: {folder}")
            if not folder_path.is_dir():
                raise NotADirectoryError(f"{folder} 不是一个目录")

        print(f"源目录验证成功: {self.source_dir}")

    def get_base_name(self, file_path, folder_name):
        base_name = file_path.stem
        # 统一处理：去掉所有文件夹文件名末尾的 _RGB 后缀
        if base_name.endswith('_RGB'):
            base_name = base_name[:-4]
        # 对 HSI 文件夹额外去掉 _resampled 后缀
        if folder_name == 'HSI' and base_name.endswith('_resampled'):
            base_name = base_name[:-10]
        return base_name

    def collect_files(self):
        folder_files = {}

        for folder in self.required_folders:
            folder_path = self.source_dir / folder
            files = set()

            if folder == 'labels':
                for subfolder in folder_path.iterdir():
                    if subfolder.is_dir():
                        for file in subfolder.iterdir():
                            if file.is_file():
                                base_name = self.get_base_name(file, folder)
                                files.add(base_name)
            else:
                for file in folder_path.iterdir():
                    if file.is_file():
                        base_name = self.get_base_name(file, folder)
                        files.add(base_name)

            folder_files[folder] = files
            print(f"{folder} 文件夹中找到 {len(files)} 个文件")

        self.common_files = set.intersection(*folder_files.values())
        print(f"找到 {len(self.common_files)} 个共有文件")

        # 新增：打印每个文件夹的前10个文件名，方便对比
        print("\n=== 各文件夹提取的文件名示例 ===")
        for folder in self.required_folders:
            print(f"\n{folder} 示例（前10个）：")
            for name in list(folder_files[folder])[:10]:
                print(f"  - {name}")

        return folder_files

    def create_target_directory(self):
        for folder in self.required_folders:
            target_folder = self.target_dir / folder
            target_folder.mkdir(parents=True, exist_ok=True)

            if folder == 'labels':
                source_labels = self.source_dir / folder
                for subfolder in source_labels.iterdir():
                    if subfolder.is_dir():
                        subfolder_name = subfolder.name
                        target_subfolder = target_folder / subfolder_name
                        target_subfolder.mkdir(exist_ok=True)

        print(f"目标目录结构创建成功: {self.target_dir}")

    def copy_file(self, source_path, target_path):
        try:
            if target_path.exists():
                if self.overwrite:
                    shutil.copy2(source_path, target_path)
                    print(f"覆盖文件: {target_path}")
                    self.processed_count += 1
                else:
                    print(f"跳过已存在文件: {target_path}")
                    self.skipped_count += 1
            else:
                shutil.copy2(source_path, target_path)
                print(f"复制文件: {source_path} -> {target_path}")
                self.processed_count += 1
        except Exception as e:
            print(f"复制文件失败: {source_path} -> {target_path}, 错误: {str(e)}")
            self.error_count += 1

    def copy_common_files(self, folder_files):
        for base_name in self.common_files:
            # 复制 HSI 文件（匹配 .tif 后缀，包括 _resampled）
            hsi_files = list((self.source_dir / 'HSI').glob(f"{base_name}*.tif"))
            print(f"\n[{base_name}] HSI 匹配到 {len(hsi_files)} 个文件")
            for hsi_file in hsi_files:
                target_hsi = self.target_dir / 'HSI' / hsi_file.name
                self.copy_file(hsi_file, target_hsi)

            # 复制 RGB 文件（匹配任意后缀，包括 _RGB）
            rgb_files = list((self.source_dir / 'RGB').glob(f"{base_name}*.*"))
            print(f"[{base_name}] RGB 匹配到 {len(rgb_files)} 个文件")
            for rgb_file in rgb_files:
                target_rgb = self.target_dir / 'RGB' / rgb_file.name
                self.copy_file(rgb_file, target_rgb)

            # 复制 IR 文件（匹配任意后缀，包括 _RGB）
            ir_files = list((self.source_dir / 'IR').glob(f"{base_name}*.*"))
            print(f"[{base_name}] IR 匹配到 {len(ir_files)} 个文件")
            for ir_file in ir_files:
                target_ir = self.target_dir / 'IR' / ir_file.name
                self.copy_file(ir_file, target_ir)

            # 复制 labels 文件（宽松匹配：base_name 包含在文件名中）
            labels_dir = self.source_dir / 'labels'
            print(f"[{base_name}] 开始匹配 labels 文件...")
            label_count = 0
            for subfolder in labels_dir.iterdir():
                if subfolder.is_dir():
                    # 用更宽松的匹配：文件名包含 base_name
                    label_files = [f for f in subfolder.iterdir() if f.is_file() and base_name in f.name]
                    label_count += len(label_files)
                    for label_file in label_files:
                        target_subfolder = self.target_dir / 'labels' / subfolder.name
                        target_label = target_subfolder / label_file.name
                        self.copy_file(label_file, target_label)
            print(f"[{base_name}] labels 匹配到 {label_count} 个文件")

    def process(self):
        try:
            self.validate_source_directory()
            folder_files = self.collect_files()
            self.create_target_directory()
            self.copy_common_files(folder_files)

            print("\n处理完成！")
            print(f"成功处理: {self.processed_count} 个文件")
            print(f"跳过: {self.skipped_count} 个文件")
            print(f"错误: {self.error_count} 个文件")
            print(f"目标目录: {self.target_dir}")

        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="提取源目录中四个子文件夹的共有文件到目标目录")
    parser.add_argument('--source', type=str, default=DEFAULT_SOURCE_DIR, help='源目录路径 (默认: %(default)s)')
    parser.add_argument('--target', type=str, default=DEFAULT_TARGET_DIR, help='目标目录路径 (默认: %(default)s)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')

    args = parser.parse_args()

    print(f"使用源目录: {args.source}")
    print(f"使用目标目录: {args.target}")

    processor = FileProcessor(args.source, args.target, args.overwrite)
    processor.process()


if __name__ == "__main__":
    main()