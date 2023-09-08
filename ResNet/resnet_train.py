from torchvision.models import resnet152, resnet50
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torchvision.transforms import transforms, ToTensor
from torchvision.datasets import CIFAR10, mnist
from tqdm import tqdm
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluate(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_label = batch[1].to(device)

            logits = model(batch_data)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            tags = batch_label.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)
    correct_num = sum(int(x == y) for (x, y) in zip(pred_tags, true_tags))
    accuracy = correct_num / len(pred_tags)

    return accuracy

def train(model, data_loader, loss_func, optimizer):
    model.train()
    for batch in tqdm(data_loader):
        data = batch[0].to(device)
        label = batch[1].to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, label)
        
        loss.backward()
        optimizer.step()
        
def getdataset(dataset_type='cifar'):
    if dataset_type == "cifar":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        torch.manual_seed(0)
        g = torch.Generator()
        return DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_train), shuffle=True, generator=g, batch_size=32, num_workers=4), \
            DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform_test),
                            batch_size=32, num_workers=4)    
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        train_dataset = mnist.MNIST(root='./mnist/train', train=True, transform=transform_train, download=True)
        test_dataset = mnist.MNIST(root='./mnist/test', train=False, transform=transform_test, download=True)
        return DataLoader(train_dataset, batch_size=32),\
                DataLoader(test_dataset, batch_size=32)
                
def evaluate(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_label = batch[1].to(device)

            logits = model(batch_data)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            tags = batch_label.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)
    correct_num = sum(int(x == y) for (x, y) in zip(pred_tags, true_tags))
    accuracy = correct_num / len(pred_tags)

    return accuracy
    
def main(args):
    if args.model == "resnet152":
        model = resnet152()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048, 10)
        model.to(device)
    else:
        model = resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048, 10)
        model.to(device)

    train_dataloader, test_dataloader = getdataset("cifar" if args.dataset == "cifar" else "mnist")
    if args.mode == "train":
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_func = nn.CrossEntropyLoss()

        max_test_acc = 0.
        for epoch in range(100):
            with open(f"./result/{args.model}_result.txt", "a", encoding='utf-8') as f:
                train(model, train_dataloader, loss_func, optimizer)

                torch.save(model.state_dict(), './models/'+args.dataset+'_'+args.model+'_last.pt')
                test_acc = evaluate(model, test_dataloader)
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    torch.save(model.state_dict(), './models/'+args.dataset+'_'+args.model+'_best.pt')
                    print("Best model saved!")

                print("epoch: {}  test_acc: {:.2f}%".format(epoch, test_acc * 100))
                f.write("{:.2f}\n".format(test_acc))
    else:
        if args.model == "resnet50":
            state = torch.load('./models/'+args.dataset+'_resnet50_last.pt')
        else:
            state = torch.load('./models/'+args.dataset+'_resnet152_last.pt')
        model.load_state_dict(state)
        model.to(device)
        
        test_acc = evaluate(model, test_dataloader)
        print(f"test acc is {test_acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose Model and Dataset for FPGA-BIT.')
    parser.add_argument('--model', type=str, required="True")
    parser.add_argument('--dataset', type=str, required="True")
    parser.add_argument('--mode', type=str, required="True")
    args = parser.parse_args()
    
    main(args)
