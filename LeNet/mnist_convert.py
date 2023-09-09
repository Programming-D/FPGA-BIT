import torch
from torchvision.transforms import transforms
from torchvision.datasets import mnist
transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
test_dataset = mnist.MNIST(root='./mnist/test', train=False, transform=transform_test, download=True)
device = 'cpu'
f1 = open("mnist/data.txt", "w", encoding="utf-8")
f2 = open("mnist/label.txt", "w", encoding="utf-8")
num = 0
for idx, (train_x, train_label) in enumerate(test_dataset):
    if num >= 100:
        break
    num += 1
    
    train_x = list(train_x.to(device)[0].flatten().numpy())
    train_label = train_label
    
    line_data = ",".join(map(str, train_x))+",\n"
    line_label = str(train_label)+","
    f1.write(line_data)
    f2.write(line_label)
            
