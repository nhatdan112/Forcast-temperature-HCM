import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import rasterio
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# Hàm để chọn file từ cửa sổ
def select_file():
    Tk().withdraw()  # Ẩn cửa sổ chính
    filename = askopenfilename(filetypes=[("GeoTIFF files", "*.tif;*.tiff")])
    return filename

lst_file = select_file()
with rasterio.open(lst_file) as src:
    lst_data = src.read(1)

# Kiểm tra và xử lý giá trị NaN
lst_data = np.where(lst_data == 0, np.nan, lst_data)  
lst_data_flattened = lst_data.flatten()
lst_data_flattened = lst_data_flattened[~np.isnan(lst_data_flattened)]  

# Giảm kích thước dữ liệu (lấy mẫu mỗi n pixel để giảm kích thước)
sample_rate = 100  # Lấy mẫu mỗi 100 pixel
lst_data_sampled = lst_data_flattened[::sample_rate]

number_of_periods = min(500, len(lst_data_sampled))  
data = pd.Series(lst_data_sampled[:number_of_periods])

# Chia dữ liệu thành dữ liệu huấn luyện và kiểm tra
train_data = data[:-20]
test_data = data[-20:]

# 1. Mô hình ARIMA
try:
    arima_model = ARIMA(train_data, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=20)
except Exception as e:
    print("Lỗi khi chạy mô hình ARIMA:", e)
    arima_forecast = np.full(20, np.nan)  # Dự báo NaN nếu lỗi xảy ra

# 2. Mô hình SVM
X = np.array(range(len(train_data))).reshape(-1, 1)
y = train_data.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(np.array(range(len(train_data), len(train_data) + 20)).reshape(-1, 1))

# 3. Mô hình LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_total, y_total = create_dataset(data_scaled, time_step)

# Chia dữ liệu thành dữ liệu huấn luyện và kiểm tra
X_train_lstm = X_total[:-20].reshape(-1, time_step, 1)
y_train_lstm = y_total[:-20]
X_test_lstm = X_total[-20:].reshape(-1, time_step, 1)
y_test_lstm = y_total[-20:]

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Thêm kiểm tra cho quá trình huấn luyện
try:
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32)
    lstm_pred = lstm_model.predict(X_test_lstm)
    lstm_pred = scaler.inverse_transform(lstm_pred)
except Exception as e:
    print("Lỗi khi chạy mô hình LSTM:", e)
    lstm_pred = np.full((20, 1), np.nan)  # Dự báo NaN nếu lỗi xảy ra

# Giới hạn dự báo nhiệt độ trong khoảng 0-50
arima_forecast = np.clip(arima_forecast, 0, 50)
svm_pred = np.clip(svm_pred, 0, 50)
lstm_pred = np.clip(lstm_pred, 0, 50)

# Kiểm tra kích thước các dự đoán
print(f"ARIMA forecast size: {arima_forecast.shape}")
print(f"SVM prediction size: {svm_pred.shape}")
print(f"LSTM prediction size: {lstm_pred.shape}")

# Khởi tạo biến cho dự đoán kết hợp
predicted_temperatures = None

# Kiểm tra và đồng bộ kích thước của các mảng
length = 20  # Số lượng dự đoán đang kiểm tra

# Đảm bảo rằng tất cả các dự đoán có chiều dài phù hợp
if arima_forecast.size == length and svm_pred.size == length and lstm_pred.size == length:
    predicted_temperatures = (arima_forecast + svm_pred + lstm_pred.flatten()) / 3
    # Giới hạn nhiệt độ dự báo kết hợp
    predicted_temperatures = np.clip(predicted_temperatures, 0, 50)
else:
    print("Kích thước của các dự đoán không khớp. Không thể kết hợp.")
    print(f"Kích thước hiện tại: ARIMA: {arima_forecast.size}, SVM: {svm_pred.size}, LSTM: {lstm_pred.flatten().size}")

# Hiển thị kết quả dưới dạng mảng
print("Dự báo nhiệt độ của từng mô hình:")
print("ARIMA:", arima_forecast)
print("SVM:", svm_pred)
print("LSTM:", lstm_pred.flatten())
if predicted_temperatures is not None:
    print("Dự báo nhiệt độ kết hợp:", predicted_temperatures)

# Hiển thị kết quả
plt.figure(figsize=(14, 7))
plt.plot(range(20), test_data.values, label='Actual Temperature', color='blue', marker='o')  # Nhiệt độ thực tế
if predicted_temperatures is not None:
    plt.plot(range(20), predicted_temperatures, label='Combined Forecast', color='orange', linestyle='--', marker='x')  # Dự báo kết hợp

# Thêm tiêu đề và các thông số cơ bản
plt.title('Temperature Forecast using ARIMA, SVM, and LSTM')
plt.legend()
plt.xlabel('Bước thời gian')
plt.ylabel('Nhiệt độ (°C)')
plt.ylim(0, 50)  # Giới hạn trục y trong khoảng 0-50 độ C

# Hiển thị biểu đồ
plt.show()

# Tính toán sai số dự báo
arima_error = test_data.values - arima_forecast
svm_error = test_data.values - svm_pred
lstm_error = test_data.values - lstm_pred.flatten()
combined_error = test_data.values - predicted_temperatures if predicted_temperatures is not None else np.nan

# Tính toán các chỉ số thống kê cho mỗi mô hình
def calculate_statistics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    return mae, rmse, r2

arima_mae, arima_rmse, arima_r2 = calculate_statistics(test_data.values, arima_forecast)
svm_mae, svm_rmse, svm_r2 = calculate_statistics(test_data.values, svm_pred)
lstm_mae, lstm_rmse, lstm_r2 = calculate_statistics(test_data.values, lstm_pred.flatten())
combined_mae, combined_rmse, combined_r2 = calculate_statistics(test_data.values, predicted_temperatures)

# Tổng hợp dữ liệu chi tiết hơn thành file CSV
output_file = os.path.splitext(lst_file)[0] + "_detailed_summary.csv"
summary_data = pd.DataFrame({
    'Actual': test_data.values,
    'ARIMA': arima_forecast,
    'SVM': svm_pred,
    'LSTM': lstm_pred.flatten(),
    'Combined': predicted_temperatures if predicted_temperatures is not None else np.nan,
    'ARIMA_Error': arima_error,
    'SVM_Error': svm_error,
    'LSTM_Error': lstm_error,
    'Combined_Error': combined_error,
    'ARIMA_MAE': [arima_mae] * len(test_data),   # Lặp lại giá trị MAE trên mỗi dòng
    'SVM_MAE': [svm_mae] * len(test_data),
    'LSTM_MAE': [lstm_mae] * len(test_data),
    'Combined_MAE': [combined_mae] * len(test_data),
    'ARIMA_RMSE': [arima_rmse] * len(test_data),
    'SVM_RMSE': [svm_rmse] * len(test_data),
    'LSTM_RMSE': [lstm_rmse] * len(test_data),
    'Combined_RMSE': [combined_rmse] * len(test_data),
    'ARIMA_R2': [arima_r2] * len(test_data),
    'SVM_R2': [svm_r2] * len(test_data),
    'LSTM_R2': [lstm_r2] * len(test_data),
    'Combined_R2': [combined_r2] * len(test_data),
})

# Lưu dữ liệu vào file CSV chi tiết
summary_data.to_csv(output_file, index=True)
print(f"Tổng hợp dữ liệu chi tiết đã được lưu tại: {output_file}")
