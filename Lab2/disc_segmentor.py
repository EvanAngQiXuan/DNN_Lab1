import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticDiscSegmenter(nn.Module):
    def __init__(self):
        super(OpticDiscSegmenter, self).__init__()
        
        # --- ENCODER (4 Trainable Layers) ---
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # --- DECODER (4 Trainable Layers) ---
        self.dec1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1) 
        self.dec2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)
        self.dec4 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        p1 = self.pool(e1)
        
        e2 = F.relu(self.enc2(p1))
        p2 = self.pool(e2)
        
        e3 = F.relu(self.enc3(p2))
        p3 = self.pool(e3)
        
        e4 = F.relu(self.enc4(p3))  # bottleneck at 1/8 resolution (no 4th pool, so skips line up)

        # Decoder with Skip Connections (Bilinear Interpolation + Concat)
        d1 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = F.relu(self.dec1(d1))
        
        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.dec2(d2))
        
        d3 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = F.relu(self.dec3(d3))
        
        out = self.dec4(d3)  # Logits output at full resolution (use BCEWithLogitsLoss or combined loss)
        
        return out