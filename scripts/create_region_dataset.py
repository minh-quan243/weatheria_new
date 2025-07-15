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
# 3. Phân vùng theo vĩ độ
# =========================
def get_region(lat):
    if lat >= 37: return "North"
    elif lat >= 33: return "Central"
    else: return "South"

df["region"] = df["lat"].apply(get_region)

# =========================
# 4. Lấy thông tin lần đầu xuất hiện mỗi bão
# =========================
df_sorted = df.sort_values(["id", "time"])
first_appearance = df_sorted.drop_duplicates("id", keep="first").copy()

first_appearance["month"] = first_appearance["time"].dt.month
first_appearance["season"] = first_appearance["month"].apply(get_season)

# =========================
# 5. Tổng hợp theo tháng và tạo cột "region"
# =========================
monthly_counts = first_appearance.groupby(["year", "month", "season"]).agg(
    region=("region", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
).reset_index()

# =========================
# 6. Tạo dữ liệu cho các tháng không có bão
# =========================
all_months = pd.DataFrame([(year, month) for year in monthly_counts['year'].unique() for month in range(1, 13)], columns=["year", "month"])
all_months["season"] = all_months["month"].apply(get_season)

# Merge để điền các tháng không có bão
merged = pd.merge(all_months, monthly_counts, on=["year", "month", "season"], how="left")

# Điền dữ liệu cho các tháng không có bão
merged["region"] = merged["region"].fillna("Unknown")  # Nếu không có bão, sẽ gán là "Unknown"

# Giữ lại chỉ các cột cần thiết: năm, tháng, mùa, region
final_df = merged[["year", "month", "season", "region"]]

# =========================
# 7. Lưu kết quả
# =========================
output_path = "D:/Pycharm/weather-new/data/Processed/Year"
os.makedirs(output_path, exist_ok=True)
final_df.to_csv(f"{output_path}/region_dataset.csv", index=False)
print("✅ Dataset đã lưu tại:", f"{output_path}/region_dataset.csv")
