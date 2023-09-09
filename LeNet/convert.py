import torch

def dec2bin(x):
    '''
    将十进制小数x转为对应的二进制小数
    '''
    x -= int(x)
    bins = []
    while x:
        x *= 2
        bins.append("1" if x >= 1. else "0")
        x -= int(x)
    return "".join(bins)


def float2IEEE16(x):
    '''
    float转IEEE754的半精度浮点数
    '''
    ms = "0" if x > 0 else "1"
    x = abs(x)
    x0 = int(x)  # 整数部分
    x1 = x - x0  # 小数部分
    x0 = bin(x0).replace("0b", "")
    x1 = dec2bin(x1)
    if x0[0] == "0":
        E = 15 - x1.find("1") - 1
        m = x1[x1.find("1"):]
        if E < 0:
            E = 15
            m = "00000000000"
    else:
        E = 15 + len(x0) - 1
        m = x0 + x1
    E = bin(E).replace("0b", "")
    if len(E) > 5:
        E = E[:5]
    else:
        for i in range(5 - len(E)):
            E = "0" + E
    m = m[1:]
    if len(m) > 10:
        m = m[:10]
    else:
        for i in range(10 - len(m)):
            m += "0"
    y = ms+E+m
    y1 = ""
    for i in range(len(y)//4):
        y1 += hex(int(y[4*i:4*(i+1)], 2)).replace("0x", "")
    return y1


def float2IEEE32(x):
    '''
    float转IEEE754的单精度浮点数
    '''
    ms = "0" if x > 0 else "1"
    x = abs(x)
    x0 = int(x)  # 整数部分
    x1 = x - x0  # 小数部分
    x0 = bin(x0).replace("0b", "")
    x1 = dec2bin(x1)
    if x0[0] == "0":
        E = 127 - x1.find("1") - 1
        m = x1[x1.find("1"):]
        if E < 0:
            E = 127
            m = "000000000000000000000000"
    else:
        E = 127 + len(x0) - 1
        m = x0 + x1
    E = bin(E).replace("0b", "")
    if len(E) > 8:
        E = E[:8]
    else:
        for i in range(8 - len(E)):
            E = "0" + E
    m = m[1:]
    if len(m) > 23:
        m = m[:23]
    else:
        for i in range(23 - len(m)):
            m += "0"
    y = ms+E+m
    y1 = ""
    for i in range(len(y)//4):
        y1 += hex(int(y[4*i:4*(i+1)], 2)).replace("0x", "")
    return y1

def floatto18f(x):
    y1 = "{:.18f}".format(x)
    return y1


if __name__ == '__main__':
    from model import Lenet5
    model = Lenet5()
    state = torch.load("models/mnist_distilled_lenet5_best.pt")
    model.load_state_dict(state)
    for name in model.state_dict():
        print(name)

        # 卷积层权重量化
        if name in ["conv1.weight", "conv2.weight", "conv3.weight"]:
            fname = name.split(".")[0]
            Tensor = model.state_dict()[name]
            s1, s2, s3, s4 = Tensor.shape
            if name == "conv2.weight":
                fname = 'conv3'
            elif name == "conv3.weight":
                fname = 'conv5'
            with open("parameters/W"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(s1):
                    for j in range(s2):
                        for k in range(s3):
                            for t in range(s4):
                                f.write(floatto18f(Tensor[i][j][k][t])+',\n')
                                
        # 全连接层权重量化
        if name in ["fc1.weight", "fc2.weight"]:
            fname = name.split(".")[0]
            Matrix = model.state_dict()[name].T
            with open("parameters/W"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(Matrix.shape[0]):
                    for j in range(Matrix.shape[1]):
                        f.write(floatto18f(Matrix[i][j])+",\n")
                        
        # 池化层权重量化
        if name in ["pool1.weight", "pool2.weight"]:
            fname = name.split(".")[0]
            Tensor = model.state_dict()[name]
            s = Tensor.shape[0]
            with open("parameters/W"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(s):
                    for j in range(4):
                        f.write(floatto18f(10*Tensor[i][j][0][0])+',\n')
        
        # 卷积层偏置量化
        if name in ["conv1.bias", "conv2.bias", "conv3.bias"]:
            fname = name.split(".")[0]
            Tensor = model.state_dict()[name]
            s = Tensor.shape[0]
            if name == "conv2.weight":
                fname = 'conv3'
            elif name == "conv3.weight":
                fname = 'conv5'
            with open("parameters/b"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(s):
                    f.write(floatto18f(Tensor[i])+',\n')
                    
        # 全连接层偏置量化
        if name in ["fc1.bias", "fc2.bias"]:
            fname = name.split(".")[0]
            Matrix = model.state_dict()[name]
            with open("parameters/b"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(Matrix.shape[0]):
                    f.write(floatto18f(Matrix[i])+",\n")
                    
        # 池化层偏置量化
        if name in ["pool1.bias", "pool2.bias"]:
            fname = name.split(".")[0]
            Tensor = model.state_dict()[name]
            s = Tensor.shape[0]
            with open("parameters/b"+fname+".h", "w", encoding="utf-8") as f:
                for i in range(s):
                    f.write(floatto18f(Tensor[i])+',\n')
                        
