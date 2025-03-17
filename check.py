import os
import rasterio
from PIL import Image

def check_shapes(image_dir, mask_dir):
    image_shapes = set()
    mask_shapes = set()
    
    # images
    for img_file in sorted(os.listdir(image_dir)):
        if img_file.endswith('.tif'):
            img_path = os.path.join(image_dir, img_file)
            with rasterio.open(img_path) as src:
                image_shapes.add((src.height, src.width))
    
    # masks
    for mask_file in sorted(os.listdir(mask_dir)):
        if mask_file.endswith('.tif'):
            mask_path = os.path.join(mask_dir, mask_file)
            with rasterio.open(mask_path) as src:
                mask_shapes.add((src.height, src.width))
    
    print(f"Unique image shapes: {image_shapes}")
    print(f"Unique mask shapes: {mask_shapes}")
    
    if len(image_shapes) > 1:
        print("Error: Images have inconsistent shapes!")
    else:
        print("All images have the same shape.")
    
    if len(mask_shapes) > 1:
        print("Error: Masks have inconsistent shapes!")
    else:
        print("All masks have the same shape.")
    
    if image_shapes and mask_shapes and image_shapes.pop() != mask_shapes.pop():
        print("Error: Image and mask shapes do not match!")
    else:
        print("All image and mask shapes match.")

if __name__ == '__main__':
    cropped_image_dir = './cropped_images'
    cropped_mask_dir = './cropped_masks'
    
    if not os.path.exists(cropped_image_dir) or not os.path.exists(cropped_mask_dir):
        print("Error: Cropped directories do not exist.")
    else:
        check_shapes(cropped_image_dir, cropped_mask_dir)
