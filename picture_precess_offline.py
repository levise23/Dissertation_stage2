import math
import cv2
import rasterio
from rasterio.windows import Window
import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Geod
from tqdm import tqdm
TIF_PATH='/usr1/home/s125mdg43_07/remote/UAV/UAV_VisLoc_dataset/13/satellite13.tif'
sate_row = {'LT_lon_map': 116.033769, 'RB_lon_map': 116.064566, 
            'LT_lat_map': 29.817376, 'RB_lat_map': 29.725402, 'width': 11482, 'height': 34291}
class Opt:
    def __init__(self):
        self.w = 300           # 切片宽度 (米)
        self.h = 300           # 切片高度 (米)
        self.alpha = 0.5       # 重叠比例 (0.1 代表 10% 重叠)
        self.output_size = (224, 224) # 模型输入分辨率 (DINOv2)

def generate_satellite_gallery(tif_path, sate_row, opt, output_gallery_dir):
    output_gallery_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 提取卫星图基础边界
    lon_min, lon_max = sate_row['LT_lon_map'], sate_row['RB_lon_map']
    lat_min, lat_max = sate_row['RB_lat_map'], sate_row['LT_lat_map']
    img_w, img_h = sate_row['width'], sate_row['height']

    # 2. 计算卫星图的绝对物理尺寸 (米)
    geod = Geod(ellps='WGS84')
    _, _, total_w_m = geod.inv(lon_min, lat_max, lon_max, lat_max) # 顶部宽
    _, _, total_h_m = geod.inv(lon_min, lat_min, lon_min, lat_max) # 左侧高
    
    # 像素分辨率 (米/像素)
    res_x = total_w_m / img_w 
    res_y = total_h_m / img_h 

    # 3. 计算步长 (米)
    stride_w_m = opt.w * (1 - opt.alpha)
    stride_h_m = opt.h * (1 - opt.alpha)

    # 4. 计算网格数
    num_x = math.ceil((total_w_m - opt.w) / stride_w_m) + 1 if total_w_m > opt.w else 1
    num_y = math.ceil((total_h_m - opt.h) / stride_h_m) + 1 if total_h_m > opt.h else 1

    print(f"🌍 卫星图总尺寸: {total_w_m:.1f}m x {total_h_m:.1f}m")
    print(f"🔪 计划切片数量: {num_x} x {num_y} = {num_x * num_y} 张")

    recorder = []
    patch_id = 0
    with rasterio.open(tif_path) as src:
        for i in tqdm(range(num_x), desc="Processing Columns"):
            for j in range(num_y):
                x_start_m = i * stride_w_m
                y_start_m = j * stride_h_m
                
                # [修复 2] 严格处理边缘对齐与图幅过小的问题
                if total_w_m >= opt.w and x_start_m + opt.w > total_w_m: 
                    x_start_m = total_w_m - opt.w
                elif total_w_m < opt.w:
                    x_start_m = 0
                    
                if total_h_m >= opt.h and y_start_m + opt.h > total_h_m: 
                    y_start_m = total_h_m - opt.h
                elif total_h_m < opt.h:
                    y_start_m = 0

                center_x_m = x_start_m + opt.w / 2.0
                center_y_m = y_start_m + opt.h / 2.0
                
                # [修复 1] 弃用线性插值，使用 WGS84 椭球体正向高精度推演经纬度
                # 先向北(方位角 0)走 y_start_m，再向东(方位角 90)走 x_start_m
                _, center_lat_temp, _ = geod.fwd(lon_min, lat_min, 0, center_y_m)
                center_lon, center_lat, _ = geod.fwd(lon_min, center_lat_temp, 90, center_x_m)

                # 计算像素坐标
                center_px_x = center_x_m / res_x
                center_px_y = (total_h_m - center_y_m) / res_y 
                
                px_width = opt.w / res_x
                px_height = opt.h / res_y

                # [修复 4] 使用 round 避免 int 截断导致的像素漂移
                col_start = round(center_px_x - px_width / 2)
                row_start = round(center_px_y - px_height / 2)

                window = Window(col_start, row_start, round(px_width), round(px_height))
                img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                img = img.transpose(1, 2, 0)
                
                # [修复 3] 防止 16-bit 遥感图像转存 JPG 变黑图
                if img.dtype == np.uint16 or img.max() > 255:
                    img = (img / img.max() * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_resized = cv2.resize(img_bgr, opt.output_size)
                
                # ... (保存图像与记录 CSV 的代码保持不变) ...
                
                patch_name = f"gallery_patch_{patch_id:06d}.jpg"
                save_path = output_gallery_dir / patch_name
                cv2.imwrite(str(save_path), img_resized)

                # 记录属性
                recorder.append({
                    'patch_id': patch_id,
                    'path': save_path,
                    'center_x_m': round(center_x_m, 2),  # 局部 X (米)
                    'center_y_m': round(center_y_m, 2),  # 局部 Y (米)
                    'center_lon': round(center_lon, 8),
                    'center_lat': round(center_lat, 8),
                    'width_m': opt.w,
                    'height_m': opt.h,
                    'is_edge': (i == num_x-1) or (j == num_y-1) # 标记是否为图幅边缘
                })
                patch_id += 1

    # 导出 CSV
    df = pd.DataFrame(recorder)
    csv_path = output_gallery_dir / "gallery.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Gallery 建库完成！共 {len(df)} 张切片，元数据保存至: {csv_path}")
    return df

# 使用示例：
# opt = Opt()
# 伪造一个 sate_row 测试数据
# sate_row = {'LT_lon_map': 115.0, 'RB_lon_map': 115.01, 'LT_lat_map': 29.01, 'RB_lat_map': 29.0, 'width': 10000, 'height': 10000}
generate_satellite_gallery(TIF_PATH, sate_row, Opt(), Path("/usr1/home/s125mdg43_07/remote/stage2dataset/gallery"))