import pandas as pd
import os

# =========================
# 1. Load dữ liệu
# =========================
file_path = "D:/Pycharm/weather-new/data/Raw/ibtracs_japan.csv"
df = pd.read_csv(file_path, parse_dates=["time"])
df = df.dropna(subset=["lat", "lon"])  # Tránh lỗi

# =========================
# 2. Thời gian và mùa
# =========================
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Autumn"

df["season"] = df["month"].apply(get_season)

# =========================
# 3. Lấy thông tin lần đầu xuất hiện mỗi bão
# =========================
df_sorted = df.sort_values(["id", "time"])
first_appearance = df_sorted.drop_duplicates("id", keep="first").copy()

first_appearance["month"] = first_appearance["time"].dt.month
first_appearance["season"] = first_appearance["month"].apply(get_season)

# =========================
# 4. Tổng hợp theo tháng và tạo cột "num_typhoons"
# =========================
monthly_counts = first_appearance.groupby(["year", "month", "season"]).agg(
    num_typhoons=("id", "count")
).reset_index()

# =========================
# 5. Tạo dữ liệu cho các tháng không có bão
# =========================
all_months = pd.DataFrame([(year, month) for year in monthly_counts['year'].unique() for month in range(1, 13)], columns=["year", "month"])
all_months["season"] = all_months["month"].apply(get_season)

# Merge để điền các tháng không có bão
merged = pd.merge(all_months, monthly_counts, on=["year", "month", "season"], how="left")

# Điền dữ liệu cho các tháng không có bão
merged["num_typhoons"] = merged["num_typhoons"].fillna(0)  # Nếu không có bão thì đặt số lượng bão là 0

# Giữ lại chỉ các cột cần thiết: năm, tháng, mùa, num_typhoons
final_df = merged[["year", "month", "season", "num_typhoons"]]

# =========================
# 6. Lưu kết quả
# =========================
output_path = "D:/Pycharm/weather-new/data/Processed/Year"
os.makedirs(output_path, exist_ok=True)
final_df.to_csv(f"{output_path}/numtyphoon_dataset.csv", index=False)
print("✅ Dataset đã lưu tại:", f"{output_path}/numtyphoon_dataset.csv")
