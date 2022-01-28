# 视觉注意力模型的分析与比较

## 实验环境

* 操作系统 Windows 11
* python 3.8.12
* 显卡 GTX1650
* CUDA 11.2
* pytorch 1.10.1+cu113
* torch_vision 0.11.2+cu113

## 环境配置方法  

* 创建环境  
conda create --name py38 python=3.8 -y  
* 激活环境  
conda activate py38  
* 安装 CUDA  
网址：<https://developer.nvidia.com/cuda-toolkit-archive>  
* 安装 pytorch  
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f <https://download.pytorch.org/whl/cu113/torch_stable.html>  
或在以下网址选择其它版本：<https://pytorch.org/get-started/locally/>  

* 安装其它依赖  
pip install d2l==0.17.3  
pip install sklearn

## 数据集

* 运行 Read_Cifar.py 获取 Cifar-100 数据集  
* 运行 Resd_Cifar_10.py 获取 cifar-10 数据集

## 运行方法

配置好环境之后运行main.py即可。
默认是选择训练模式。  
也可以选择我的预训练模型进行测试。

## 文件说明

| 文件名 | 内容 |
| ------ | -------- |
| Accumulator.py | 计算准确率的计数器 |
| CA_SA.py | CBAM中的两种注意力机制 |
| LeNet_5.py | LeNet5 |
| main.py | 主函数，定义模型，训练和测试 |
| Pre_CA_SA.py | <u> CBAM轻量版“改进” </u> |
| ProgressBar.py | 用于显示训练进度 |
| Read_Cifar.py | Cifar-100下载、预处理、分割 |
| Read_Cifar.py | Cifar-10下载、预处理、分割 |
| Residual.py | 传统版残差块 |
| Residual_CBAM.py | 加入CBAM的残差块 |
| Residual_CBAM_Pro.py | 加入CBAM_Pro的残差块 |
| Residual_ECA.py | 加入ECA的残差块 |
| Residual_SE.py | 加入SE的残差块 |
| ResNet18.py | 传统ResNet18 |
| ResNet18_CBAM.py | 加入CBAM的ResNet18 |
| ResNet18_CBAM_Pro.py | 加入CBAM_Pro的ResNet18 |
| ResNet18_ECA.py | 加入ECA的ResNet18 |
| ResNet18_SE.py | 加入SE的ResNet18 |
| ResNet101.py | 传统ResNet101 |
| ResNet101_CBAM.py | 加入CBAM的ResNet101 |
| ResNet101_ECA.py | 加入ECA的ResNet101 |
| ResNet101_SE.py | 加入SE的ResNet101 |

如果想要运行技术报告中表2架构的ResNet，需要将对应网络的第一个卷积层参数改为:

self.b1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  
                   nn.BatchNorm2d(64), nn.ReLU(),  
                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1))  

## 实验结果

### On cifar-100

| Method | Test Acc |
| ------ | -------- |
| ResNet-18 | 0.52 |
| ResNet-18-SE | 0.54 |
| ResNet-18-ECA | 0.56 |
| ResNet-18-CBAM | 0.55 |
| ResNet-18-CBAM-Pro | 0.54 |
| ResNet-18*-CBAM-Pro | 0.60 |

### On cifar-10

| Method | Test Acc |
| ------ | -------- |
| ResNet-18| 0.84 |
| ResNet-18-SE| 0.85 |
| ResNet-18-ECA| 0.84 |
| ResNet-18-CBAM| 0.84 |
| ResNet-18-CBAM-Pro| 0.84 |
| ResNet-18*-CBAM-Pro| 0.88 |

其中*表示使用技术报告中表2架构的ResNet-18
