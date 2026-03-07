"""
Utility script to copy images to results directory
"""

from pathlib import Path
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def copy_image_to_results(image_path, results_dir="results", rename=None):
    """
    读取指定位置的图片，并保存到results文件夹
    
    Args:
        image_path (str): 图片的完整路径或相对路径
        results_dir (str): 结果保存的目录，默认为"results"
        rename (str): 可选，新的文件名。如果为None，使用原文件名
    
    Returns:
        str: 保存后的图片完整路径
    
    Example:
        >>> copy_image_to_results(
        ...     "/egr/research-actionlab/caizhon2/codes/EQA/3DSPI/spatial_planning/"
        ...     "datasets/evaluation/SAT/./data/train/image_132118_0.png"
        ... )
        'results/image_132118_0.png'
    """
    image_path = Path(image_path)
    
    # 检查文件是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    # 创建results目录
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # 决定输出文件名
    output_name = rename if rename else image_path.name
    output_path = results_path / output_name
    
    # 使用PIL读取并保存图片
    try:
        img = Image.open(image_path)
        img.save(output_path)
        logger.info(f"✓ 图片已保存: {output_path}")
    except Exception as e:
        logger.error(f"✗ 保存失败: {e}")
        raise
    
    return str(output_path)


def copy_images_batch(image_paths, results_dir="results"):
    """
    批量复制多张图片到results文件夹
    
    Args:
        image_paths (list): 图片路径列表
        results_dir (str): 结果保存的目录，默认为"results"
    
    Returns:
        list: 保存后的图片路径列表
    
    Example:
        >>> paths = [
        ...     "/path/to/image1.png",
        ...     "/path/to/image2.png",
        ... ]
        >>> copy_images_batch(paths)
        ['results/image1.png', 'results/image2.png']
    """
    output_paths = []
    for i, img_path in enumerate(image_paths, 1):
        try:
            result = copy_image_to_results(img_path, results_dir)
            output_paths.append(result)
        except Exception as e:
            logger.error(f"[{i}/{len(image_paths)}] 处理失败: {img_path} - {e}")
    
    logger.info(f"完成: {len(output_paths)}/{len(image_paths)} 图片已保存")
    return output_paths


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行使用: python copy_images.py <image_path> [output_dir]
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
        
        result = copy_image_to_results(image_path, output_dir)
        print(f"Saved to: {result}")
    else:
        print("Usage: python copy_images.py <image_path> [output_dir]")
        print("Example:")
        print("  python copy_images.py /path/to/image.png results")
