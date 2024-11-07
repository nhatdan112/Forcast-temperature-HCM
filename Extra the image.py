import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename


# Đọc metadata từ file MTL.txt
def read_metadata(mtl_file):
    metadata = {}
    with open(mtl_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split(' = ')
                metadata[key] = value.strip('"')
    return metadata

# Chuyển đổi giá trị số DN thành bức xạ
def dn_to_radiance(dn, ml, al):
    return ml * dn + al

# Chuyển đổi bức xạ thành nhiệt độ sáng (Brightness Temperature)
def radiance_to_bt(radiance, k1, k2):
    # Tránh giá trị vô hạn hoặc không xác định
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = k2 / (np.log((k1 / radiance) + 1))
    return bt

# Hiệu chỉnh nhiệt độ bề mặt với hệ số phát xạ
def bt_to_lst(bt, emissivity, wavelength=10.8e-6, rho=1.438e-2):
    return bt / (1 + (wavelength * bt / rho) * np.log(emissivity))

# Chuyển đổi Kelvin sang Celsius
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# Hàm mở hộp thoại chọn file
def select_file(prompt):
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ gốc
    file_path = askopenfilename(title=prompt)
    return file_path

# Hàm mở hộp thoại lưu file
def save_file(prompt):
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ gốc
    file_path = asksaveasfilename(title=prompt, defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif")])
    return file_path

# Mở cửa sổ chọn file thay vì nhập tay
band_10_file = select_file("Chọn file Band 10 (.tif): ")
mtl_file = select_file("Chọn file MTL.txt: ")
output_file = save_file("Chọn nơi lưu file kết quả LST (GeoTIFF): ")

# Đảm bảo đường dẫn file được chọn
band_10_file = r"{}".format(band_10_file)
mtl_file = r"{}".format(mtl_file)
output_file = r"{}".format(output_file)

# Đọc metadata
metadata = read_metadata(mtl_file)

# Các hệ số từ metadata
ml_band_10 = float(metadata['RADIANCE_MULT_BAND_10'])  # Hệ số ML cho Band 10
al_band_10 = float(metadata['RADIANCE_ADD_BAND_10'])   # Hệ số AL cho Band 10
k1_constant = float(metadata['K1_CONSTANT_BAND_10'])   # Hằng số K1
k2_constant = float(metadata['K2_CONSTANT_BAND_10'])   # Hằng số K2

# Đọc băng nhiệt (Band 10)
with rasterio.open(band_10_file) as src:
    band_10 = src.read(1)
    # Thay thế giá trị 0 bằng NaN để loại bỏ chúng trong phân tích
    band_10 = np.where(band_10 == 0, np.nan, band_10)
    profile = src.profile  # Lưu thông tin về định dạng file để xuất file sau này

# Chuyển đổi DN thành bức xạ
radiance = dn_to_radiance(band_10, ml_band_10, al_band_10)

# In giá trị min, max bức xạ
print(f"Min radiance value: {np.nanmin(radiance)}")
print(f"Max radiance value: {np.nanmax(radiance)}")

# Chuyển đổi bức xạ thành nhiệt độ sáng (BT - Brightness Temperature)
bt = radiance_to_bt(radiance, k1_constant, k2_constant)

# Kiểm tra giá trị min, max BT
print(f"Min BT value: {np.nanmin(bt)}")
print(f"Max BT value: {np.nanmax(bt)}")

# Giả sử hệ số phát xạ (emissivity) là 0.98
emissivity = 0.98

# Chuyển đổi nhiệt độ sáng thành LST (Land Surface Temperature)
lst_kelvin = bt_to_lst(bt, emissivity)
lst_kelvin = bt_to_lst(bt, emissivity)
# Kiểm tra nếu giá trị LST âm, thì thiết lập thành NaN
lst_kelvin[lst_kelvin < 0] = np.nan  
lst_celsius = kelvin_to_celsius(lst_kelvin)

# Chuyển đổi từ Kelvin sang Celsius
lst_celsius = kelvin_to_celsius(lst_kelvin)

# Kiểm tra giá trị min, max LST và loại bỏ NaN
lst_celsius = np.where(np.isnan(lst_celsius), 0, lst_celsius)  # Thay NaN bằng 0 cho việc hiển thị
min_lst = np.nanmin(lst_celsius)
max_lst = np.nanmax(lst_celsius)

# Hiển thị bản đồ nhiệt độ bề mặt
plt.figure(figsize=(10, 6))
plt.title('Land Surface Temperature (Celsius)')
img = plt.imshow(lst_celsius, cmap='RdYlBu_r', vmin=min_lst, vmax=max_lst)  # Thiết lập giá trị min, max cho hiển thị
plt.colorbar(img, label='Temperature (°C)')
plt.show()

# Cập nhật profile cho output (định dạng GeoTIFF)
profile.update(dtype=rasterio.float32, count=1, compress='lzw')

# Xuất file kết quả LST dưới dạng GeoTIFF
try:
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(lst_celsius.astype(rasterio.float32), 1)
    print(f"Kết quả LST đã được lưu tại: {output_file}")
except Exception as e:
    print(f"Đã xảy ra lỗi khi lưu file: {e}")