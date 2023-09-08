from model import Lenet5
from torchvision.models import resnet152, resnet50
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, mnist
from tqdm.contrib import tzip
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
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        transform_student = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4734), (0.2507)),
        ])
        torch.manual_seed(0)
        g = torch.Generator()
        return DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_train, download=True), shuffle=True, generator=g, batch_size=32, num_workers=4), \
            DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform_test, download=True),
                            batch_size=32, num_workers=4), \
            DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_student, download=True),
                        shuffle=True, generator=g, batch_size=32, num_workers=4)
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
                DataLoader(test_dataset, batch_size=32), \
                DataLoader(train_dataset, batch_size=32),

def run(args, EPOCHS, lr, batch_size, alpha, T):
    if args.model == "resnet152":
        teacher_model = resnet152()
        teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        teacher_model.fc = nn.Linear(2048, 10)
        teacher_model.to(device)
        teacher_model.load_state_dict(torch.load('../ResNet/models/'+args.dataset+'_resnet152_best.pt'))
    else:
        teacher_model = resnet50()
        teacher_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        teacher_model.fc = nn.Linear(2048, 10)
        teacher_model.to(device)
        teacher_model.load_state_dict(torch.load('../ResNet/models/'+args.dataset+'_resnet50_best.pt'))

    train_dataloader_teacher, test_dataloader, train_dataloader_student = getdataset("cifar" if args.dataset == "cifar" else "mnist")
    student_model = Lenet5().to(device)
    if args.mode == "train":
        optimizer = Adam(student_model.parameters(), lr=lr)
        student_loss = nn.CrossEntropyLoss()
        distill_loss = nn.KLDivLoss(reduction='batchmean')

        teacher_model.eval()
        max_test_acc = 0.
        for epoch in range(EPOCHS):
            student_model.train()
            for (batch_tea, batch_stu) in tzip(train_dataloader_teacher, train_dataloader_student):
                tea_data = batch_tea[0].to(device)
                stu_data = batch_stu[0].to(device)
                stu_label = batch_stu[1].to(device)

                with torch.no_grad():
                    teacher_logits = teacher_model(tea_data)

                student_model.zero_grad()
                logits = student_model(stu_data)
                loss = alpha * student_loss(logits, stu_label) + (1 - alpha) * distill_loss(
                    F.log_softmax(logits / T, dim=1), F.softmax(teacher_logits / T, dim=1))

                loss.backward()
                optimizer.step()

            torch.save(student_model.state_dict(), './models/'+args.dataset+'_distilled_lenet5_last.pt')

            test_acc = evaluate(student_model, test_dataloader)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(student_model.state_dict(), './models/'+args.dataset+'_distilled_lenet5_best.pt')
                print("Best model saved!")

            print("epoch: {}  test_acc: {:.2f}%".format(epoch, test_acc * 100))
    else:
        student_model.load_state_dict(torch.load('./models/'+args.dataset+'_distilled_lenet5_best.pt'))
        test_acc = evaluate(student_model, test_dataloader)
        print("test_acc: {:.2f}%".format(test_acc * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose Model and Dataset for FPGA-BIT ditilling.')
    parser.add_argument('--model', type=str, required="True")
    parser.add_argument('--dataset', type=str, required="True")
    parser.add_argument('--mode', type=str, required="True")
    args = parser.parse_args()
    run(args, EPOCHS=100, lr=0.001, batch_size=128, alpha=0.3, T=7)
