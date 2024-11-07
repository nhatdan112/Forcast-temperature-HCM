import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog
from rasterio.mask import mask

def select_directory(title="Select a directory"):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    print(f"Directory selected: {folder_path}")
    return folder_path

def select_shapefile(title="Select a shapefile"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Shapefiles", "*.shp")])
    print(f"Shapefile selected: {file_path}")
    return file_path

def merge_images(folder1, folder2, band_name='B10'):
    print("Starting image merging process for B10 band...")
    datasets = []
    for folder in [folder1, folder2]:
        for filename in filter(lambda f: f.lower().endswith('.tif') and band_name in f, os.listdir(folder)):
            image_path = os.path.join(folder, filename)
            print(f"Opening image: {filename}")
            src = rasterio.open(image_path)
            datasets.append(src)
    
    if not datasets:
        raise ValueError("No images corresponding to the specified band were found.")

    mosaic, out_trans = merge(datasets)
    print("Images successfully merged.")
    meta = datasets[0].meta.copy()
    meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })

    for ds in datasets:
        ds.close()

    return mosaic, out_trans, meta

def merge_mtl_data(folder1, folder2):
    print("Merging metadata from .mtl files...")
    mtl_files = [os.path.join(folder, f) for folder in [folder1, folder2] for f in os.listdir(folder) if f.endswith('.txt')]
    mtl_data = {}
    for mtl_file in mtl_files:
        print(f"Reading metadata file: {mtl_file}")
        with open(mtl_file, 'r') as file:
            for line in file:
                print(f"Processing line: {line.strip()}")  # Debug line
                if "=" in line:
                    key, value = line.strip().split('=')
                    key, value = key.strip(), value.strip()
                    if key in mtl_data:
                        key = f"{key}_{os.path.basename(mtl_file)}"  # Append filename to avoid duplication
                    mtl_data[key] = value
    print("Metadata merged successfully.")
    return mtl_data


def save_mtl_data(mtl_data, save_path):
    print(f"Saving merged metadata to {save_path}")
    with open(save_path, 'w') as file:
        for key, value in mtl_data.items():
            file.write(f"{key} = {value}\n")
    print("Metadata saved successfully as a .txt file.")

def cut_image_by_shapefile(mosaic, out_trans, meta, shapefile_path):
    print("Starting the image cropping process by shapefile...")
    geo = gpd.read_file(shapefile_path)
    if geo.crs != meta['crs']:
        print("Transforming GeoDataFrame to the raster's CRS...")
        geo = geo.to_crs(meta['crs'])

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            # Check if the mosaic is three-dimensional and correct it
            if mosaic.ndim == 3 and mosaic.shape[0] == 1:
                mosaic = mosaic.squeeze()  # Reduce the first dimension if it's singleband
            dataset.write(mosaic, 1)
            out_image, out_transform = mask(dataset, geo.geometry, crop=True, all_touched=True)
            out_meta = meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            return out_image, out_meta

def main():
    folder1 = select_directory("Select the first image directory:")
    folder2 = select_directory("Select the second image directory:")
    shapefile_path = select_shapefile("Select the shapefile for cropping:")
    mosaic, out_trans, meta = merge_images(folder1, folder2)
    mtl_data = merge_mtl_data(folder1, folder2)
    cropped_image, cropped_meta = cut_image_by_shapefile(mosaic, out_trans, meta, shapefile_path)

    # Check and correct the shape of cropped_image before saving
    if cropped_image is not None:
        # If cropped_image has an unnecessary first dimension, squeeze it out
        if cropped_image.ndim == 3 and cropped_image.shape[0] == 1:
            cropped_image = cropped_image.squeeze(0)  # Squeeze the first dimension

        save_folder = select_directory("Select folder to save the cropped image and metadata:")
        image_save_path = os.path.join(save_folder, 'cropped_image.tif')
        mtl_save_path = os.path.join(save_folder, 'merged_metadata.txt')  # Change to .txt format
        
        # Update metadata to match the cropped image's dimensions
        cropped_meta.update({
            "height": cropped_image.shape[0],
            "width": cropped_image.shape[1],
            "transform": cropped_meta['transform']
        })
        
        # Write the cropped image to a new file
        with rasterio.open(image_save_path, "w", **cropped_meta) as dest:
            dest.write(cropped_image, 1)  # Ensure cropped_image is 2D for single-band

        # Save merged metadata as a .txt file
        save_mtl_data(mtl_data, mtl_save_path)
        print(f"Image and metadata saved to {save_folder}")

if __name__ == "__main__":
    main()
