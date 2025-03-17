from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
import os
import pickle

def crop_save(image_path, mask_path, savedir1, savedir2, croph, cropw, ratw, rath, alpha):
    # Use rasterio to open the image and mask
    with rasterio.open(image_path) as src_image, rasterio.open(mask_path) as src_mask:
        image = src_image.read()
        mask = src_mask.read()
        
        width, height = src_image.width, src_image.height
        
        if croph >= height or cropw >= width:
            print(f"Error: Crop size cannot be larger than the original image: {image_path}")
            return False
        elif ratw >= height or ratw >= width:
            print(f"Error: Window sliding step is larger than image dimensions: {image_path}")
            return False
        elif rath >= height or rath >= width:
            print(f"Error: Window sliding step is larger than image dimensions: {image_path}")
            return False
        else:
            count = 1
            x = alpha
            x_i = 0
            x_list = []
            y_list = []
            while x <= width - ratw - 92:
                y = alpha
                y_i = 0
                while y <= height - rath - 92:
                    # Use rasterio's Window for cropping
                    window = Window(x, y, cropw, croph)
                    new_img = src_image.read(window=window)
                    new_mask = src_mask.read(window=window)
                    # print(new_img.shape)
                    # print(new_mask.shape)
                    
                    # Save the cropped image and mask
                    if not os.path.exists(savedir1):
                        print("Selected folder does not exist, trying to create it.")
                        os.makedirs(savedir1)
                    if not os.path.exists(savedir2):
                        print("Selected folder does not exist, trying to create it.")
                        os.makedirs(savedir2)
                    
                    # Save image
                    output_image_path = os.path.join(savedir1, f"{os.path.splitext(os.path.basename(image_path))[0]}_pstrips_{count}.tif")
                    with rasterio.open(output_image_path, 'w', 
                                      driver='GTiff', width=cropw, height=croph, 
                                      count=src_image.count, dtype=src_image.dtypes[0]) as dst:
                        dst.write(new_img)
                    
                    # Save mask
                    output_mask_path = os.path.join(savedir2, f"{os.path.splitext(os.path.basename(mask_path))[0]}_pstrips_{count}.tif")
                    with rasterio.open(output_mask_path, 'w', 
                                      driver='GTiff', width=cropw, height=croph, 
                                      count=src_image.count, dtype=src_image.dtypes[0]) as dst:
                        dst.write(new_mask)
                    
                    count += 1
                    x_list.append(x_i)
                    y_list.append(y_i)
                    y += rath
                    y_i += 1
                x += ratw
                x_i += 1
            list_f = list(zip(x_list, y_list))
            return list_f

def process_folder(raw_data_dir, raw_labels_dir, savediroimg, savedirmask, croph, cropw, ratw, rath, alpha):
    # Get all files from raw_data and raw_labels directories
    image_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(raw_labels_dir) if f.endswith('.png')])
    
    # Ensure that the number of images and masks match
    if len(image_files) != len(mask_files):
        print("Error: The number of image files does not match the number of mask files!")
        return
    
    all_coordinates = []
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(raw_data_dir, img_file)
        mask_path = os.path.join(raw_labels_dir, mask_file)
        
        print(f"Processing: {img_file} and {mask_file}")
        coordinates = crop_save(img_path, mask_path, savediroimg, savedirmask, croph, cropw, ratw, rath, alpha)
        if coordinates:
            all_coordinates.extend(coordinates)
    
    # Save all coordinate list
    with open(os.path.join(savediroimg, 'point_list.pkl'), 'wb') as f:
        pickle.dump(all_coordinates, f)

if __name__ == '__main__':
    # Input folder paths
    raw_data_dir = r'./raw_data'  # Folder containing .tif files
    raw_labels_dir = r'./raw_labels'  # Folder containing .png files
    savediroimg = r'./cropped_images'  # Folder to save cropped images
    savedirmask = r'./cropped_masks'  # Folder to save cropped masks
    
    # Cropping parameters
    croph, cropw = 252, 252  # Crop height and width
    ratw, rath = 68, 68  # Sliding window steps
    alpha = -92  # Starting offset
    
    # Process all files in the folder
    process_folder(raw_data_dir, raw_labels_dir, savediroimg, savedirmask, croph, cropw, ratw, rath, alpha)
 
