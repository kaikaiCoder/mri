import torch
import torch.nn as nn
import monai

class PredictionModule(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # 使用MONAI的UNet3D作为分割backbone
        self.segmentation_net = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=embedding_dim,
            out_channels=2,  # 二分类分割
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
        
        # 分类头
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 各个预测任务的输出层
        self.idh_predictor = nn.Linear(64, 2)
        self.p19q_predictor = nn.Linear(64, 2)
        self.mgmt_predictor = nn.Linear(64, 2)
        
    def forward(self, embeddings):
        # 分割预测
        segmentation = self.segmentation_net(embeddings)
        
        # 特征提���用于分类
        features = self.classification_head(embeddings)
        
        # 多任务预测
        idh_pred = self.idh_predictor(features)
        p19q_pred = self.p19q_predictor(features)
        mgmt_pred = self.mgmt_predictor(features)
        
        return {
            'segmentation': segmentation,
            'idh': idh_pred,
            '1p19q': p19q_pred,
            'mgmt': mgmt_pred
        } 