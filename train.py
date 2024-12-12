import torch
from torch.utils.data import DataLoader
from models.contrastive_module import ContrastiveLearningModule
from models.prediction_module import PredictionModule
from data.dataset import MultiModalMRIDataset
from utils.losses import compute_seg_loss, compute_cls_loss
from config import Config

def train():
    # 初始化数据集
    dataset = MultiModalMRIDataset(data_dir=Config.data_dir, patch_size=Config.patch_size)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 初始化模型
    contrastive_model = ContrastiveLearningModule(
        input_channels=Config.input_channels, 
        embedding_dim=Config.embedding_dim
    ).to(Config.device)
    prediction_model = PredictionModule(
        embedding_dim=Config.embedding_dim
    ).to(Config.device)
    
    # 优化器
    contrastive_optimizer = torch.optim.Adam(
        contrastive_model.parameters(), 
        lr=Config.learning_rate
    )
    prediction_optimizer = torch.optim.Adam(
        prediction_model.parameters(), 
        lr=Config.learning_rate
    )
    
    # 训练循环
    for epoch in range(Config.num_epochs):
        for batch in dataloader:
            # 将数据移到指定设备
            patches = batch['patches'].to(Config.device)
            
            # 第一阶段：对比学习
            embeddings = contrastive_model(patches)
            contrastive_loss = contrastive_model.contrastive_loss(embeddings, embeddings)
            
            contrastive_optimizer.zero_grad()
            contrastive_loss.backward()
            contrastive_optimizer.step()
            
            # 第二阶段：预测
            with torch.no_grad():
                embeddings = contrastive_model(patches)
            
            predictions = prediction_model(embeddings)
            
            # 计算多任务损失
            seg_loss = compute_seg_loss(predictions['segmentation'], batch['tumor_mask'])
            idh_loss = compute_cls_loss(predictions['idh'], batch['idh'])
            p19q_loss = compute_cls_loss(predictions['1p19q'], batch['1p19q'])
            mgmt_loss = compute_cls_loss(predictions['mgmt'], batch['mgmt'])
            
            # 计算加权多任务损失
            total_loss = (
                Config.seg_loss_weight * seg_loss +
                Config.idh_loss_weight * idh_loss +
                Config.p19q_loss_weight * p19q_loss +
                Config.mgmt_loss_weight * mgmt_loss
            )
            
            prediction_optimizer.zero_grad()
            total_loss.backward()
            prediction_optimizer.step()

if __name__ == '__main__':
    train() 