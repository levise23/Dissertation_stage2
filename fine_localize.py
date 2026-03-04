import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from kornia.feature import LoFTR

# ==========================================
# 1. 核心参数修正
# ==========================================
# 物理长度 300m / 图像宽度 224px
METERS_PER_PIXEL = 300.0 / 224.0  
INPUT_SIZE = 640  # LoFTR 内部推理尺寸

def fine_match():
    if not os.path.exists('match_pairs_for_fine.csv'):
        print("❌ 找不到输入文件")
        return

    df = pd.read_csv('match_pairs_for_fine.csv')
    matcher = LoFTR(pretrained='outdoor').cuda().eval()
    results = []

    print(f"[*] 正在处理 {len(df)} 对图像 (分辨率: {METERS_PER_PIXEL:.4f} m/px)")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        uav_p, sate_p = row['uav_path'], row['sate_path']
        if not os.path.exists(uav_p) or not os.path.exists(sate_p): continue

        img0_raw = cv2.imread(uav_p, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(sate_p, cv2.IMREAD_GRAYSCALE)
        
        # 记录原始尺寸 (应该是 224)
        h1, w1 = img1_raw.shape 

        # LoFTR 推理
        t0 = torch.from_numpy(cv2.resize(img0_raw, (INPUT_SIZE, INPUT_SIZE))).float()[None, None].cuda() / 255.
        t1 = torch.from_numpy(cv2.resize(img1_raw, (INPUT_SIZE, INPUT_SIZE))).float()[None, None].cuda() / 255.

        with torch.no_grad():
            out = matcher({"image0": t0, "image1": t1})
            mkpts0, mkpts1 = out['keypoints0'].cpu().numpy(), out['keypoints1'].cpu().numpy()

        fine_x, fine_y = row['retrieved_x'], row['retrieved_y']
        
        # 只有匹配点足够多才进行修正
        if len(mkpts0) > 15:
            M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if M is not None:
                # 计算 UAV 中心在 640 尺度下的投影
                u_c_640 = np.array([[[INPUT_SIZE/2, INPUT_SIZE/2]]], dtype=np.float32)
                res_640 = cv2.perspectiveTransform(u_c_640, M)[0][0]
                
                # 映射回 224 原始像素尺寸
                u_raw = res_640[0] * (w1 / INPUT_SIZE)
                v_raw = res_640[1] * (h1 / INPUT_SIZE)
                
                # 相对于切片中心的偏移 (像素)
                du = u_raw - (w1 / 2)
                dv = v_raw - (h1 / 2)
                
                # 转换为地理坐标偏移
                fine_x = row['retrieved_x'] + du * METERS_PER_PIXEL
                fine_y = row['retrieved_y'] - dv * METERS_PER_PIXEL # V向下为正，Y向上为正，故用减法

        results.append({
            'fine_x': fine_x, 'fine_y': fine_y,
            'gt_x': row['gt_x'], 'gt_y': row['gt_y'],
            'coarse_error': np.sqrt((row['retrieved_x']-row['gt_x'])**2 + (row['retrieved_y']-row['gt_y'])**2),
            'fine_error': np.sqrt((fine_x-row['gt_x'])**2 + (fine_y-row['gt_y'])**2)
        })

    # 保存并分析结果
    res_df = pd.DataFrame(results)
    res_df.to_csv('final_fine_localization_results.csv', index=False)

    # 【重要】分段统计逻辑
    # 我们只看粗检索正确的样本（误差 < 150m），因为错误的样本算精配准没意义
    success_mask = res_df['coarse_error'] < 150
    valid_results = res_df[success_mask]

    print("\n" + "="*50)
    print(f"📊 全样本平均误差 (包含失败的 42%): {res_df['fine_error'].mean():.2f} 米")
    print("-" * 50)
    print(f"🎯 粗检索命中样本分析 (误差 < 150m, 共 {len(valid_results)} 帧):")
    print(f"命中样本 - 粗定位平均误差: {valid_results['coarse_error'].mean():.2f} 米")
    print(f"命中样本 - 精配准平均误差: {valid_results['fine_error'].mean():.2f} 米")
    print(f"精度提升: {(valid_results['coarse_error'].mean() - valid_results['fine_error'].mean()):.2f} 米")
    print("="*50)

if __name__ == "__main__":
    fine_match()