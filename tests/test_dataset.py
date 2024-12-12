import unittest
import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
import tempfile
import os
from data.dataset import MultiModalMRIDataset

class TestMultiModalMRIDataset(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # 创建测试用的临床数据CSV
        self.clinical_data = pd.DataFrame({
            'ID': ['UCSF-PDGM-001', 'UCSF-PDGM-002'],
            'Sex': ['F', 'M'],
            'Age': [67, 45],
            'Grade': [4, 3],
            'IDH': [0, 1],
            'MGMT': [1.0, 0.0],
            '1p/19q': [0.0, 1.0]
        })
        self.csv_path = self.data_dir / 'clinical_info.csv'
        self.clinical_data.to_csv(self.csv_path, index=False)
        
        # 创建测试用的MRI数据
        self.create_test_data()
        
    def create_test_data(self):
        # 创建模拟的MRI数据
        for id in self.clinical_data['ID']:
            # 创建随机数据 (240, 240, 155, 4)
            mri_data = np.random.rand(240, 240, 155, 4).astype(np.float32)
            
            # 保存为pkl文件
            pkl_path = self.data_dir / f"{id}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(mri_data, f)
                
    def test_dataset_loading(self):
        dataset = MultiModalMRIDataset(
            data_dir=self.data_dir,
            clinical_csv=self.csv_path,
            patch_size=64
        )
        
        # 测试数据集大小
        self.assertEqual(len(dataset), 2)
        
        # 测试单个样本
        sample = dataset[0]
        
        # 验证返回的字典键
        expected_keys = {'patches', 'idh', '1p19q', 'mgmt'}
        self.assertEqual(set(sample.keys()), expected_keys)
        
        # 验证patches的形状
        patches = sample['patches']
        self.assertIsInstance(patches, torch.Tensor)
        self.assertEqual(len(patches.shape), 5)  # (N, 4, 64, 64, 64)
        self.assertEqual(patches.shape[1], 4)    # 4个模态
        self.assertEqual(patches.shape[2:], (64, 64, 64))  # patch大小
        
        # 验证标签类型
        self.assertIsInstance(sample['idh'], (int, np.integer))
        self.assertIsInstance(sample['1p19q'], (float, np.floating))
        self.assertIsInstance(sample['mgmt'], (float, np.floating))
        
    def test_patch_extraction(self):
        dataset = MultiModalMRIDataset(
            data_dir=self.data_dir,
            clinical_csv=self.csv_path,
            patch_size=64
        )
        
        # 测试patch提取
        sample = dataset[0]
        patches = sample['patches']
        
        # 计算期望的patch数量
        x_steps = (240 - 64) // (64//2) + 1
        y_steps = (240 - 64) // (64//2) + 1
        z_steps = (155 - 64) // (64//2) + 1
        expected_patches = x_steps * y_steps * z_steps
        
        self.assertEqual(patches.shape[0], expected_patches)
        
    def tearDown(self):
        # 清理临时文件
        for file in self.data_dir.glob('*'):
            file.unlink()
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 