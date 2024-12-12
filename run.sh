#!/bin/bash

# 激活虚拟环境
source mri_env/bin/activate

# 设置环境变量
export $(cat .env | xargs)

# 运行测试脚本
python test_env.py