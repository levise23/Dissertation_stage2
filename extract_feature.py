import os
import torch
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast # 补齐对齐的精度转换
from torch.utils.data import Dataset, DataLoader
import numpy as np
# 导入你自己的模型构建函数
from models.model import make_model 

def get_parse():
    parser = argparse.ArgumentParser(description='Offline Feature Extraction')
    parser.add_argument('--csv_path', type=str,default='/usr1/home/s125mdg43_07/remote/stage2dataset/gallery/gallery.csv', help='建库生成的 CSV 文件路径 (须包含 path)')
    parser.add_argument('--output_pt', type=str,default='/usr1/home/s125mdg43_07/remote/stage2dataset/sate02res.pt' , help='输出的 .pt 特征库保存路径')
    parser.add_argument('--weight_path', type=str, default='/usr1/home/s125mdg43_07/remote/rebuild/Dissertation/checkpoints/curriculum_s2_o1/net_020.pth', help='训练好的模型权重文件路径 (.pth)')
    
    # 架构与数据参数
    #parser.add_argument('--nclasses', default=701, type=int, help='训练时的总类别数')
    parser.add_argument('--block', default=3, type=int, help='热力分组数')
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--triplet_loss', default=0.4, type=float, help='保持 >0')
    
    # 【新增】工程提速参数
    parser.add_argument('--batchsize', default=128, type=int, help='提取特征的 Batch Size')
    parser.add_argument('--num_workers', default=8, type=int, help='多进程读取数量')
    parser.add_argument('--filename', default='path')
    return parser.parse_args()

# ==========================================
# 专门为离线建库写的轻量级 Dataset
# ==========================================
class GalleryDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, img_path
        except Exception as e:
            # 万一有坏图，返回一个全黑张量保底，防止 dataloader 崩溃
            print(f"\n[Warning] 读取图片失败 {img_path}: {e}")
            return torch.zeros((3, 224, 224)), img_path

def load_custom_model(opt, weight_path, device):
    print(f"[*] 正在解析权重文件: {weight_path}")
    
    # 1. 先把权重字典拉进内存
    state_dict = torch.load(weight_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'net_dict' in state_dict:
        state_dict = state_dict['net_dict']
    
    # 清理 'module.' 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '') if k.startswith('module.') else k
        clean_state_dict[clean_key] = v

    # ==========================================
    # 2. 动态侦测分类数 (黑魔法在这里)
    # ==========================================
    classifier_key = 'global_classifier.classifier.weight'
    if classifier_key in clean_state_dict:
        # 提取权重矩阵的第 0 维，这就是训练时的真实类别数！
        detected_nclasses = clean_state_dict[classifier_key].shape[0]
        print(f"[✓] 自动侦测到该模型训练时的总类别数 (nclasses) 为: {detected_nclasses}")
        opt.nclasses = detected_nclasses  # 强行覆盖 opt 里的假数据
    else:
        print(f"[!] 警告: 权重中未找到全局分类头 '{classifier_key}'，将使用默认类别数。")

    # 3. 此时 opt.nclasses 已经是 100% 正确的了，放心初始化模型
    print(f"[*] 正在初始化模型骨架 (block: {opt.block})...")
    model = make_model(opt)
    
    # 4. 严丝合缝地加载权重
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval() 
    model.to(device)
    print("[✓] 模型及权重完美加载完毕！\n")
    
    return model
def fsra_normalize(ff):
        """FSRA 原版3D特征归一化: L2-norm on dim=1, scale by sqrt(num_parts), flatten"""
        if len(ff.shape) == 3:
            # [B, 512, block+1]
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(ff.size(-1))
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)  # [B, 512*(block+1)]
        else:
            # [B, 512] 单 block 退化情况
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        return ff
@torch.no_grad()
def extract_single_view_batch(model, img_batch):
    """
    评估模式下 model(x1, x2) 直接返回 (y1, y2)，
    每个 y 的形状为 [B, 512, block+1]，已完成局部+全局拼接。
    直接取 y1 做 fsra_normalize 即可。
    """
    # eval 模式: 返回 (y1, y2)，y1 形状 [B, 512, block+1]
    y1, _ = model(img_batch, img_batch)
    
    # 应用 fsra_normalize (L2-norm on dim=1, scale by sqrt(block+1), flatten)
    final_feat = fsra_normalize(y1)
    
    return final_feat

def main():
    opt = get_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(opt.csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {opt.csv_path}")
    df = pd.read_csv(opt.csv_path)
    
    if opt.filename not in df.columns:
        raise ValueError("CSV中找不到 'path' 列！")
        
    image_paths = df['path'].unique().tolist()
    print(f"[*] 共发现 {len(image_paths)} 张待提取图片，Batch Size: {opt.batchsize}")

    data_transform = transforms.Compose([
        transforms.Resize(size=(opt.h, opt.w), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 【新增】使用 DataLoader 拉满 IO 和 GPU 并行计算
    dataset = GalleryDataset(image_paths, data_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batchsize, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True # 锁页内存，加速 CPU 到 GPU 的数据搬运
    )

    model = load_custom_model(opt, opt.weight_path, device)
    features_list = []
    
    # 进度条基于 batch
    for img_batch, _ in tqdm(dataloader, desc="批量提取离线特征"):
        img_batch = img_batch.to(device)
        
        # 【关键修复】加上 autocast，严格对齐验证集的特征流形
        with torch.amp.autocast('cuda'):
            feat_batch = extract_single_view_batch(model, img_batch)
            
        # L2 归一化后存入 CPU 内存，防止 GPU 显存爆掉
        feat_batch = F.normalize(feat_batch, p=2, dim=1)
        features_list.append(feat_batch.cpu())

    if features_list:
        gallery_features = torch.cat(features_list, dim=0) # [N, 768]
        Path(opt.output_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gallery_features, opt.output_pt)
        
        print(f"\n✅ 特征提取完毕！")
        print(f"📊 特征矩阵维度: {gallery_features.shape}")
        print(f"💾 已保存至: {opt.output_pt}")
    else:
        print("\n❌ 未能成功提取任何特征！")

if __name__ == "__main__":
    main()