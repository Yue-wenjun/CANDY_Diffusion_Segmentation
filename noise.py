import h5py
import numpy as np
import os
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import shutil
from tqdm import tqdm


def decode_filenames(h5_file):
    """解码文件名数组"""
    try:
        # 读取文件名数据
        filenames_data = h5_file['filenames'][:]  # 形状 (31, 464)

        # 转换为字符串
        decoded_filenames = []
        for i in range(filenames_data.shape[1]):  # 遍历464列
            # 获取每个文件的字符编码
            chars = filenames_data[:, i]
            # 转换为字符串（忽略0值）
            filename = ''.join(chr(c) for c in chars if c != 0)
            decoded_filenames.append(filename)

        print(f"成功解码 {len(decoded_filenames)} 个文件名")
        return decoded_filenames

    except Exception as e:
        print(f"解码文件名时出错: {e}")
        return None


def create_noised_dataset(raw_data_dir, noise_mat_file, output_dir, snr_db=20):
    """
    创建加噪数据集

    参数:
    - raw_data_dir: 原始图像目录
    - noise_mat_file: .mat噪声文件
    - output_dir: 输出目录 (如 'noised_data_20dB')
    - snr_db: SNR值 (用于创建目录名)
    """

    # 创建输出目录
    output_dir_name = f"noised_data_{snr_db}dB"
    output_path = os.path.join(output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"=== 开始创建加噪数据集 ===")
    print(f"原始数据目录: {raw_data_dir}")
    print(f"噪声文件: {noise_mat_file}")
    print(f"输出目录: {output_path}")
    print(f"SNR: {snr_db} dB")

    # 获取原始图像文件列表
    image_extensions = ['.tif', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(raw_data_dir).glob(f'*{ext}')))

    image_files = sorted([str(f) for f in image_files])
    print(f"找到 {len(image_files)} 个原始图像文件")

    # 读取噪声数据
    print(f"\n=== 读取噪声数据 ===")
    try:
        with h5py.File(noise_mat_file, 'r') as f:
            # 检查噪声数据集
            noise_group = f['image_Nos']
            noise_keys = sorted(noise_group.keys(),
                                key=lambda x: int(x.split('_')[-1]))

            print(f"噪声数据集中有 {len(noise_keys)} 个噪声图像")

            # 解码文件名（如果有）
            decoded_filenames = decode_filenames(f)

            # 读取SNR信息
            if 'SNR_in_dB' in f:
                target_snr = f['SNR_in_dB'][0][0]
                print(f"目标SNR: {target_snr} dB")

            if 'generated_SNRs_in_dB' in f:
                actual_snrs = f['generated_SNRs_in_dB'][:]
                print(f"生成的实际SNR范围: {actual_snrs.min():.2f} - {actual_snrs.max():.2f} dB")

            # 开始处理每个图像
            print(f"\n=== 开始加噪处理 ===")

            # 如果文件名解码成功，使用文件名映射
            if decoded_filenames and len(decoded_filenames) == len(image_files):
                print("使用文件名映射进行匹配...")
                processed_count = 0

                for i, orig_filename in enumerate(tqdm(image_files, desc="加噪处理")):
                    orig_basename = os.path.basename(orig_filename)

                    # 在解码的文件名列表中查找匹配
                    matched_idx = -1
                    for j, decoded_name in enumerate(decoded_filenames):
                        # 简单的文件名匹配（可以根据需要调整）
                        if decoded_name in orig_basename or orig_basename in decoded_name:
                            matched_idx = j
                            break

                    if matched_idx >= 0:
                        # 处理匹配的图像
                        noise_key = f'image_{matched_idx + 1:03d}'
                        if noise_key in noise_group:
                            process_image_with_noise(
                                orig_filename,
                                noise_group[noise_key],
                                output_path,
                                snr_db
                            )
                            processed_count += 1
                        else:
                            print(f"警告: {noise_key} 不在噪声数据集中")
                    else:
                        print(f"警告: 未找到 {orig_basename} 的噪声匹配")

                print(f"成功处理 {processed_count}/{len(image_files)} 个图像")

            else:
                # 如果没有文件名映射，按顺序匹配
                print("按顺序匹配图像和噪声...")

                # 确保数量匹配
                min_count = min(len(image_files), len(noise_keys))
                print(f"将处理 {min_count} 个图像")

                for i in tqdm(range(min_count), desc="加噪处理"):
                    orig_filename = image_files[i]
                    noise_key = f'image_{i + 1:03d}'

                    if noise_key in noise_group:
                        process_image_with_noise(
                            orig_filename,
                            noise_group[noise_key],
                            output_path,
                            snr_db
                        )
                    else:
                        print(f"警告: {noise_key} 不在噪声数据集中")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== 处理完成 ===")
    print(f"加噪后的图像已保存到: {output_path}")


def process_image_with_noise(image_path, noise_dataset, output_dir, snr_db):
    """
    处理单个图像：添加噪声并保存
    """
    try:
        # 读取原始图像
        with rasterio.open(image_path) as src:
            # 读取图像数据
            image_data = src.read()  # 形状 (1, height, width)

            # 获取元数据
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

            # 获取图像的高度和宽度
            height, width = image_data.shape[1], image_data.shape[2]

        # 读取噪声数据
        noise_data = noise_dataset[:]  # 形状 (height, width)
        noise_data = noise_data.T

        # 检查形状是否匹配
        if noise_data.shape[0] != height or noise_data.shape[1] != width:
            print(f"形状不匹配: 图像 {height}x{width}, 噪声 {noise_data.shape}")

            # 尝试调整噪声大小以匹配图像
            # 这里使用简单的裁剪或填充（根据实际情况调整）
            if noise_data.shape[0] >= height and noise_data.shape[1] >= width:
                # 裁剪噪声以匹配图像
                noise_data = noise_data[:height, :width]
            elif noise_data.shape[0] <= height and noise_data.shape[1] <= width:
                # 用零填充噪声（根据需求调整填充方式）
                padded_noise = np.zeros((height, width), dtype=noise_data.dtype)
                padded_noise[:noise_data.shape[0], :noise_data.shape[1]] = noise_data
                noise_data = padded_noise
            else:
                print(f"无法调整噪声大小: {noise_data.shape} -> {height}x{width}")
                return

        # 添加噪声到图像
        # 注意：图像数据形状是 (1, H, W)，噪声是 (H, W)
        noised_image = image_data.copy()

        # 处理NaN值
        if np.isnan(noised_image).any():
            # 用0替换NaN
            noised_image = np.nan_to_num(noised_image, nan=0.0)

        # 添加噪声（广播噪声到图像的第一个波段）
        noised_image[0] = noised_image[0] + noise_data

        # 创建输出文件名
        orig_basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(orig_basename)[0]
        output_filename = f"{name_without_ext}_noised_{snr_db}dB.tif"
        output_path = os.path.join(output_dir, output_filename)

        # 保存加噪后的图像
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,  # 单波段
                dtype=noised_image.dtype,
                crs=crs,
                transform=transform,
                compress='lzw'  # 使用压缩以减少文件大小
        ) as dst:
            dst.write(noised_image)

        # 计算并显示一些统计信息（可选）
        if np.random.random() < 0.05:  # 随机显示5%的图像的统计信息
            print(f"  {orig_basename}: 原始范围 [{image_data.min():.3f}, {image_data.max():.3f}], "
                  f"加噪后 [{noised_image.min():.3f}, {noised_image.max():.3f}]")

        return True

    except Exception as e:
        print(f"处理图像 {os.path.basename(image_path)} 时出错: {e}")
        return False


def verify_noise_addition(noised_dir, original_dir, num_samples=5):
    """
    验证加噪效果
    """
    print(f"\n=== 验证加噪结果 ===")

    noised_files = sorted([f for f in os.listdir(noised_dir) if f.endswith('.tif')])
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.tif')])

    print(f"加噪目录文件数: {len(noised_files)}")
    print(f"原始目录文件数: {len(original_files)}")

    # 检查几个样本
    for i in range(min(num_samples, len(noised_files))):
        noised_file = noised_files[i]
        # 找到对应的原始文件
        orig_name = noised_file.replace(f'_noised_{os.path.basename(noised_dir).split("_")[-1]}', '')
        orig_name = orig_name.replace('_noised', '')

        orig_path = os.path.join(original_dir, orig_name)
        noised_path = os.path.join(noised_dir, noised_file)

        if os.path.exists(orig_path):
            try:
                with rasterio.open(orig_path) as src_orig:
                    orig_data = src_orig.read()

                with rasterio.open(noised_path) as src_noised:
                    noised_data = src_noised.read()

                # 计算差异
                diff = noised_data - orig_data
                diff_nonzero = diff[diff != 0]

                print(f"\n样本 {i + 1}: {orig_name}")
                print(f"  原始形状: {orig_data.shape}, 范围: [{orig_data.min():.3f}, {orig_data.max():.3f}]")
                print(f"  加噪形状: {noised_data.shape}, 范围: [{noised_data.min():.3f}, {noised_data.max():.3f}]")
                print(f"  差异统计: 非零点数={len(diff_nonzero)}, "
                      f"平均差异={diff.mean():.6f}, 最大差异={diff.max():.6f}")

            except Exception as e:
                print(f"验证样本 {orig_name} 时出错: {e}")
        else:
            print(f"未找到原始文件: {orig_name}")


# 主函数
if __name__ == '__main__':
    # 配置参数
    RAW_DATA_DIR = 'raw_data'  # 原始图像目录
    NOISE_MAT_FILE = 'noise/Gamma_noise_SNR_var_definition_20.mat'  # 噪声文件
    OUTPUT_BASE_DIR = '.'  # 输出基础目录
    SNR_DB = 20  # SNR值，从文件名提取或手动指定

    # 从文件名提取SNR值
    import re

    match = re.search(r'(\d+)dB', NOISE_MAT_FILE)
    if match:
        SNR_DB = int(match.group(1))

    print(f"配置信息:")
    print(f"  原始数据目录: {RAW_DATA_DIR}")
    print(f"  噪声文件: {NOISE_MAT_FILE}")
    print(f"  SNR: {SNR_DB} dB")

    # 检查目录和文件
    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误: 原始数据目录不存在: {RAW_DATA_DIR}")
        exit(1)

    if not os.path.exists(NOISE_MAT_FILE):
        print(f"错误: 噪声文件不存在: {NOISE_MAT_FILE}")
        exit(1)

    # 创建加噪数据集
    create_noised_dataset(
        raw_data_dir=RAW_DATA_DIR,
        noise_mat_file=NOISE_MAT_FILE,
        output_dir=OUTPUT_BASE_DIR,
        snr_db=SNR_DB
    )

    # 验证结果
    output_dir_name = f"noised_data_{SNR_DB}dB"
    output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)

    if os.path.exists(output_path):
        verify_noise_addition(output_path, RAW_DATA_DIR)

        print(f"\n=== 总结 ===")
        noised_files = [f for f in os.listdir(output_path) if f.endswith('.tif')]
        print(f"成功生成 {len(noised_files)} 个加噪图像")
        print(f"输出目录: {output_path}")
    else:
        print(f"错误: 输出目录未创建: {output_path}")