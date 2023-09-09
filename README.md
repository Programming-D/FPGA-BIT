# FPGA-BIT

## 介绍

这是北京理工大学人工智能专业2023年夏季小学期“基于FPGA与深度学习的图像识别”课程项目软件部分的仓库。包括模型训练、知识蒸馏以及模型量化等内容。

## Requirements

* Python 3.8
* Pytorch 1.10

## 使用方法
将仓库克隆（或下载压缩包并解压）到本地，并进入目录

```
git clone https://github.com/Programming-D/FPGA-BIT.git
cd FPGA-BIT
```
运行脚本构建需要的目录
```
bash ./preprocess.sh
```
安装配置好anaconda，并根据yaml文件配置好环境
```
conda env create -f ./fpga_bit.yml
```
### 模型训练
为了得到知识蒸馏过程中的教师模型，我们首先需要训练好一个ResNet152或者ResNet50，同时需要指定当前训练过程中用到的数据集
```
cd ResNet 
python resnet_train.py --model resnet152(or resnet50) --dataset cifar(or mnist) --mode tarin
```
为了验证模型的效果，我们可以改变参数将其变为测试模式
```
python resnet_train.py --model resnet152(or resnet50) --dataset cifar(or mnist) --mode test
```
### 知识蒸馏
上一步我们已经得到了教师模型，现在可以直接进行蒸馏了
```
cd ../LeNet
python distill.py --model resnet152(or resnet50) --dataset cifar(or mnist) --mode tarin
```
为了验证模型的效果，我们可以改变参数将其变为测试模式
```
python distill.py --model resnet152(or resnet50) --dataset cifar(or mnist) --mode test
```
### 量化
我们在项目中也尝试了利用Pytorch自带的量化函数进行量化
```
python train.py
```
当然实际中我们用到的模型参数与数据集的导出方法如下
```
python convert.py
cd ..
python cifar_convert.py
```
### 量化后得到的模型参数与图像数据
可以从以下百度网盘链接中获取
```
链接：https://pan.baidu.com/s/1cgfByeLErKhb_pKvuOSp3A?pwd=bitN 
提取码：bitN
```
## List of contributors

* [Eason Leo](https://github.com/lyccyl1)
* [Enue Lee](https://github.com/Programming-D)
* [Z-Luan](https://github.com/Z-Luan)
* [wu-han-lin](https://github.com/wu-han-lin)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.
