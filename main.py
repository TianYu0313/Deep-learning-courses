import torch
from torch import nn
import torchvision as tv
from d2l import torch as d2l
import matplotlib.pyplot as plt
from Accumulator import Accumulator
from multiprocessing import freeze_support
from ProgressBar import ProgressBar
from Read_Cifar import cifar100_dataset
from Read_Cifar_10 import cifar10_dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



from LeNet_5 import LeNet_5
from ResNet18 import ResNet18
from ResNet18_SE import ResNet18_SE
from ResNet18_ECA import ResNet18_ECA
from ResNet18_CBAM import ResNet18_CBAM
from ResNet18_CBAM_Pro import ResNet18_CBAM_Pro
from ResNet101 import ResNet101
from ResNet101_SE import ResNet101_SE
from ResNet101_CBAM import ResNet101_CBAM
from ResNet101_ECA import ResNet101_ECA


# 计算准确率
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 验证
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    print(metric[0] / metric[1])
    return metric[0] / metric[1]


# 更改学习率
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/10


# 训练
def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    plt.figure()
    timer, num_batches = d2l.Timer(), len(train_iter)

    Loss = []
    Train_Acc = []
    Valid_acc = []
    max_valid_acc = 0
    base_root = "./ResNet18_CBAM"
    train_acc = 0
    valid_acc = 0

    for epoch in range(num_epochs):
        if epoch % 10 ==0 and epoch != 0:
            adjust_learning_rate(optimizer)
        # 训练损失之和，训练准确率之和，范例数
        metric = Accumulator(3)
        net.train()
        pbar = ProgressBar(num_batches)
        print("epoch {a} training".format(a=epoch+1))
        for i, (X, y) in enumerate(train_iter):
            pbar.update(i)
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        valid_acc = evaluate_accuracy_gpu(net, test_iter)
        Loss.append(train_l)
        Train_Acc.append(train_acc)
        Valid_acc.append(valid_acc)
        print("epoch {a}, train_loss = {b}, train_acc = {c}, valid_acc = {d}".format(a=epoch+1,
                                                                                    b=train_l,
                                                                                    c=train_acc,
                                                                                    d=valid_acc))
        print(f'{metric[2]*(epoch+1) / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(net, base_root+"/ResNet18_CBAM-epoch{b}-valAcc{a}.pkl".format(b=epoch, a=valid_acc))

    plt.plot(range(len(Train_Acc)), Train_Acc)
    plt.plot(range(len(Valid_acc)), Valid_acc)
    plt.plot(range(len(Loss)), Loss)
    plt.xlabel("epoch")
    plt.legend(['Train_Acc', 'valid_acc', 'Loss'])
    f = open(base_root+"/ResNet18_CBAM.txt", 'a')
    f.write("ResNet18_CBAM_cifar10-723")
    f.write(str(Train_Acc) + "\n")
    f.write(str(Loss) + "\n")
    f.write(str(Valid_acc) + "\n")
    f.write(str(metric[2] * num_epochs / timer.sum()) + "\n")
    f.close()
    plt.savefig(base_root+"/ResNet18_CBAM.jpg")

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'valid_acc {valid_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# 测试
def test(net, test_iter, device):
    print('testing on', device)
    net.to(device)
    metric = Accumulator(2)
    net.train()
    timer, num_batches = d2l.Timer(), len(test_iter)
    pbar = ProgressBar(num_batches)
    with torch.no_grad():
        net.eval()
        pre = torch.tensor([]).to(device)
        label = torch.tensor([]).to(device)
        for i, (X, y) in enumerate(test_iter):
            pbar.update(i)
            timer.start()
            X = X.to(device)
            y = y.to(device)
            y_hat = d2l.argmax(net(X), axis=1)
            pre = torch.cat([pre, y_hat])
            label = torch.cat([label, y])
        label = label.cpu()
        pre = pre.cpu()
        print(classification_report(label, pre))
        cfm = confusion_matrix(label, pre)
        plt.matshow(cfm, cmap=plt.cm.gray)
        plt.show()


if __name__ == "__main__":
    freeze_support()
    # 此处定义模型,如下是定义示例。ResNet101需要告知块数量
    # net = ResNet18_CBAM()
    # net = ResNet101_CBAM([3, 4, 23, 3])
    net = ResNet18_CBAM_Pro()
    batch_size = 64
    train_iter, valid_iter, test_iter = cifar10_dataset(batch_size)
    lr, num_epochs = 0.01, 50
    # 可以选择查看网络架构
    # print(net.parameters)
    train(net, train_iter, valid_iter, num_epochs, lr, device=d2l.try_gpu())

    # 如果测试，可以加载预训练模型
    # net = torch.load("C:\\tju\\21221\\神经网络\\课程设计\\code\\ResNet18_CBAM\\ResNet18-CBAMPro-cifar10.pkl")
    # test(net, test_iter, d2l.try_gpu())


