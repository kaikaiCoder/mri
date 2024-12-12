import torch
import monai
import nibabel as nib
import numpy as np

def test_environment():
    print("Python环境测试:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"MONAI版本: {monai.__version__}")
    print(f"Nibabel版本: {nib.__version__}")
    print(f"Numpy版本: {np.__version__}")

if __name__ == "__main__":
    test_environment() 