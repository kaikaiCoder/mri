import torch

class Config:
    # 数据相关
    data_dir = 'path/to/data'
    patch_size = 64
    batch_size = 8
    
    # 训练相关
    num_epochs = 100
    learning_rate = 1e-4
    
    # 模型相关
    input_channels = 4  # T1, T1CE, T2, FLAIR
    embedding_dim = 256
    
    # 损失函数权重
    seg_loss_weight = 1.0
    idh_loss_weight = 1.0
    p19q_loss_weight = 1.0
    mgmt_loss_weight = 1.0
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 