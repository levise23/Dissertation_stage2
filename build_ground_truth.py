"""
build_ground_truth.py
根据 test_pairs.csv 中无人机的 GPS 坐标，判断其落在 gallery_metadata.csv
的哪个 500×500m 切片中，生成真值表 (ground truth CSV)。

坐标映射原理:
  satellite13.tif 的边界 (来自 coordinates_with_all_info.csv):
    LT (西北角): lat=29.817376, lon=116.033769
    RB (东南角): lat=29.725402, lon=116.064566
  gallery_metadata 的 center_x_m / center_y_m 以 SW 角 (RB_lat, LT_lon) 为原点，
    x_m 向东 (经度增大), y_m 向北 (纬度增大)。

  米/度换算直接从 gallery_metadata 中两个已知切片反算，避免近似误差。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# 路径配置
# ==============================
TEST_PAIRS_CSV   = '/usr1/home/s125mdg43_07/remote/rebuild_UAV/test_pairs.csv'
GALLERY_META_CSV = '/usr1/home/s125mdg43_07/remote/stage2dataset/gallery/gallery.csv'
OUTPUT_CSV       = '/usr1/home/s125mdg43_07/remote/stage2dataset/ground_truth.csv'

# ==============================
# satellite13 边界信息
# ==============================
LT_LAT = 29.817376   # 西北角纬度 (最大纬度)
LT_LON = 116.033769  # 西北角经度 (最小经度)
RB_LAT = 29.725402   # 东南角纬度 (最小纬度) → 原点纬度
RB_LON = 116.064566  # 东南角经度 (最大经度)
# 原点 = SW 角 = (RB_LAT, LT_LON)
ORIGIN_LAT = RB_LAT
ORIGIN_LON = LT_LON
HALF=150.0
# ==============================
# 从 gallery_metadata 反算米/度换算系数
# (用至少两个非边缘切片，精度更高)
# ==============================
def compute_meter_per_degree(gallery_df):
    """利用 gallery_metadata 中的两对切片反算经纬度到米的换算系数"""
    # 经度方向: 取 x_m 不同、y_m 相同的两个切片
    lon_pairs = gallery_df[gallery_df['is_edge'] == False].sort_values('center_x_m')
    unique_x = lon_pairs['center_x_m'].unique()
    if len(unique_x) >= 2:
        g1 = lon_pairs[lon_pairs['center_x_m'] == unique_x[0]].iloc[0]
        g2 = lon_pairs[lon_pairs['center_x_m'] == unique_x[1]].iloc[0]
        m_per_deg_lon = (g2['center_x_m'] - g1['center_x_m']) / (g2['center_lon'] - g1['center_lon'])
    else:
        # fallback: 标准换算 (lat ≈ 29.77°)
        m_per_deg_lon = 111320 * np.cos(np.radians(29.77))

    # 纬度方向: 取 y_m 不同、x_m 相同的两个切片
    lat_pairs = gallery_df[gallery_df['is_edge'] == False].sort_values('center_y_m')
    unique_y = lat_pairs['center_y_m'].unique()
    if len(unique_y) >= 2:
        g1 = lat_pairs[lat_pairs['center_y_m'] == unique_y[0]].iloc[0]
        g2 = lat_pairs[lat_pairs['center_y_m'] == unique_y[1]].iloc[0]
        m_per_deg_lat = (g2['center_y_m'] - g1['center_y_m']) / (g2['center_lat'] - g1['center_lat'])
    else:
        m_per_deg_lat = 111320.0

    print(f"[*] 换算系数: {m_per_deg_lon:.1f} m/deg_lon,  {m_per_deg_lat:.1f} m/deg_lat")
    return m_per_deg_lon, m_per_deg_lat


def gps_to_m(lat, lon, m_per_deg_lat, m_per_deg_lon):
    """GPS → gallery 坐标系下的米坐标 (x_m, y_m)"""
    x_m = (lon - ORIGIN_LON) * m_per_deg_lon
    y_m = (lat - ORIGIN_LAT) * m_per_deg_lat
    return x_m, y_m


def find_patch(x_m, y_m, gallery_df, half=HALF):
    """
    返回包含 (x_m, y_m) 的切片行。
    每个切片覆盖 [center-250, center+250] 米。
    若坐标落在多张切片交界处（理论上不重叠，但用 <= 容忍数值），取距离最近的。
    若无匹配则返回 None。
    """
    dx = np.abs(gallery_df['center_x_m'].values - x_m)
    dy = np.abs(gallery_df['center_y_m'].values - y_m)
    mask = (dx <= half) & (dy <= half)
    candidates = gallery_df[mask]
    if candidates.empty:
        return None
    # 万一多个候选，取欧氏距离最近的
    dist = np.hypot(
        candidates['center_x_m'].values - x_m,
        candidates['center_y_m'].values - y_m
    )
    return candidates.iloc[int(np.argmin(dist))]


def main():
    print("[*] 读取数据...")
    gallery_df = pd.read_csv(GALLERY_META_CSV)
    test_df    = pd.read_csv(TEST_PAIRS_CSV)

    print(f"    gallery patches : {len(gallery_df)}")
    print(f"    drone queries   : {len(test_df)}")

    # 换算系数
    m_per_deg_lon, m_per_deg_lat = compute_meter_per_degree(gallery_df)

    # ==============================
    # 逐行匹配
    # ==============================
    results = []
    no_match = 0

    for _, row in test_df.iterrows():
        drone_lat = float(row['drone_lat'])
        drone_lon = float(row['drone_lon'])

        x_m, y_m = gps_to_m(drone_lat, drone_lon, m_per_deg_lat, m_per_deg_lon)
        patch = find_patch(x_m, y_m, gallery_df)

        if patch is None:
            patch_id   = -1
            patch_path = ''
            patch_lon  = np.nan
            patch_lat  = np.nan
            no_match += 1
        else:
            patch_id   = int(patch['patch_id'])
            patch_path = patch['path']
            patch_lon  = patch['center_lon']
            patch_lat  = patch['center_lat']

        results.append({
            'drone_img'      : row['drone_img'],
            'drone_path'     : row['path'],
            'drone_lat'      : drone_lat,
            'drone_lon'      : drone_lon,
            'drone_x_m'      : round(x_m, 2),
            'drone_y_m'      : round(y_m, 2),
            'gt_patch_id'    : patch_id,
            'gt_patch_path'  : patch_path,
            'gt_patch_lon'   : patch_lon,
            'gt_patch_lat'   : patch_lat,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    # ==============================
    # 统计报告
    # ==============================
    matched = len(result_df[result_df['gt_patch_id'] >= 0])
    print(f"\n✅ 真值表生成完毕！")
    print(f"   匹配成功 : {matched} / {len(result_df)}")
    print(f"   未匹配   : {no_match}  (超出 gallery 覆盖范围或边界外)")
    print(f"   涉及切片 : {result_df['gt_patch_id'].nunique() - (1 if no_match > 0 else 0)} 个不同 patch")
    print(f"💾 已保存至: {OUTPUT_CSV}")

    # 预览前5行
    print("\n--- 预览 (前5行) ---")
    preview_cols = ['drone_img', 'drone_lat', 'drone_lon', 'gt_patch_id', 'gt_patch_lat', 'gt_patch_lon']
    print(result_df[preview_cols].head().to_string(index=False))


if __name__ == '__main__':
    main()
