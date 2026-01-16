import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

class Config:
    data_path = "/root/test/VLA/openvla/libero_dpo_train.hdf5"
    save_path = "best_reward_model.pth"
    key_chosen_imgs = 'chosen_images'
    key_chosen_acts = 'chosen_actions'
    key_rejected_imgs = 'rejected_images'
    key_rejected_acts = 'rejected_actions'
    action_dim = 7
    vision_feature_dim = 512
    hidden_dim = 256
    batch_size = 32           
    num_workers = 2           # 降低进程数，HDF5 并发读取更稳
    lr = 1e-4
    epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class VisionActionRewardModel(nn.Module):
    def __init__(self, action_dim, vision_feature_dim, hidden_dim):
        super().__init__()
        # 兼容性处理
        try:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            resnet = models.resnet18(pretrained=True)
            
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.vision_encoder.parameters():
            param.requires_grad = False  # 冻结视觉，只练 MLP

        self.action_encoder = nn.Linear(action_dim, 64)
        input_dim = vision_feature_dim + 64 
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, image, action):
        v_feat = torch.flatten(self.vision_encoder(image), 1)
        a_feat = F.relu(self.action_encoder(action))
        return self.mlp(torch.cat([v_feat, a_feat], dim=1))

class VisionActionPreferenceDataset(Dataset):
    def __init__(self, hdf5_path, cfg, transform=None):
        self.hdf5_path = hdf5_path
        self.cfg = cfg
        self.transform = transform
        self.index_map = []
        
        print(f"⏳ Indexing HDF5 (Memory Efficient)...")
        with h5py.File(hdf5_path, 'r') as f:
            for key in tqdm(sorted(f.keys()), desc="Scanning"):
                # 只存 Key 和 Index，不存图片数组！
                L = min(f[key][cfg.key_chosen_acts].shape[0], 
                        f[key][cfg.key_rejected_acts].shape[0])
                for i in range(L):
                    self.index_map.append((key, i))
        print(f"✅ Indexed {len(self.index_map)} pairs.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        key, i = self.index_map[idx]
        
        # 在这里打开文件：针对多进程 DataLoader 最安全的做法
        with h5py.File(self.hdf5_path, 'r') as f:
            grp = f[key]
            c_img = self.transform(Image.fromarray(grp[self.cfg.key_chosen_imgs][i], 'RGB'))
            c_act = torch.tensor(grp[self.cfg.key_chosen_acts][i], dtype=torch.float32)
            r_img = self.transform(Image.fromarray(grp[self.cfg.key_rejected_imgs][i], 'RGB'))
            r_act = torch.tensor(grp[self.cfg.key_rejected_acts][i], dtype=torch.float32)

        return {"c_img": c_img, "c_act": c_act, "r_img": r_img, "r_act": r_act}

def train():
    cfg = Config()
    model = VisionActionRewardModel(cfg.action_dim, cfg.vision_feature_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    
    dataset = VisionActionPreferenceDataset(cfg.data_path, cfg, cfg.transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            c_img, c_act = batch['c_img'].to(cfg.device), batch['c_act'].to(cfg.device)
            r_img, r_act = batch['r_img'].to(cfg.device), batch['r_act'].to(cfg.device)
            
            optimizer.zero_grad()
            s_c, s_r = model(c_img, c_act), model(r_img, r_act)
            loss = -F.logsigmoid(s_c - s_r).mean()
            loss.backward()
            optimizer.step()
            
            correct += (s_c > s_r).sum().item()
            total += c_img.size(0)
            pbar.set_postfix({"Acc": f"{correct/total:.2%}"})
            
        epoch_acc = correct / total
        print(f"Epoch {epoch} | Acc: {epoch_acc:.2%}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), cfg.save_path)

if __name__ == "__main__":
    train()