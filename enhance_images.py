import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm
from UnderwaterEnhancement import underwater_image_enhancement


def enhance_images(input_dir, output_dir_pca, output_dir_average):
    if not os.path.exists(output_dir_pca):
        os.makedirs(output_dir_pca)
    if not os.path.exists(output_dir_average):
        os.makedirs(output_dir_average)
    
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    for image_path in tqdm(image_paths, desc="Enhancing images"):
        image = Image.open(image_path).convert("RGB")
        pcafused, averagefused = underwater_image_enhancement(image, image, 0)
        
        # 生成输出文件路径
        base_fname = os.path.basename(image_path)
        pca_output_path = os.path.join(output_dir_pca, base_fname)
        average_output_path = os.path.join(output_dir_average, base_fname)
        
        # 保存增强后的图像
        pcafused.save(pca_output_path)
        averagefused.save(average_output_path) 

if __name__ == "__main__":
    input_dir = "D:/Study/Ai/sklearn/Exercise_final/testimages/test_images"  # 输入图片目录
    output_dir_pca = "D:/Study/Ai/sklearn/Exercise_final/testimages/testimages_pca_enhance"  # PCA融合方法增强后的图片目录
    output_dir_average = "D:/Study/Ai/sklearn/Exercise_final/testimages/testimages_avg_enhance"  # 平均融合方法增强后的图片目录

    enhance_images(input_dir, output_dir_pca, output_dir_average)
