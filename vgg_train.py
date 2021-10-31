import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from vgg import VGG

#---------------------#
#    settings         #
#---------------------#
import argparse

# 获取超参数
parser = argparse.ArgumentParser(description='traing paremeters')
parser.add_argument('--start_epoch',default=0, type=int, help='start')
parser.add_argument('--path', help='resume path')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--epoch', default=20,type=int, help='total epoches')
parser.add_argument('--batch_size', default=128, help='batch size')
args = parser.parse_args()
print(args)

#---------------------#
#      MNIST          #
#---------------------#

print('-----Preparing data-----')
# 图像预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10(root='/home/zhaiyize/models/vgg/data',
                                 train=True,
                                 transform=transform_train,
                                 download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = datasets.CIFAR10(root='/home/zhaiyize/models/vgg/data',
                                 train=False,
                                 transform=transform_test,
                                 download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=120,
                         shuffle=True)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


from tqdm import tqdm
import os
if "CUDA_VISIBLE_DEVICE" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICE"]="0"

# 判断是否重新训练
if args.start_epoch > 0:
    print('-----resuming from logs------')
    state = torch.load(args.path)
    net = state['net']
    best_acc = state['acc']
else:
    print('-----build new model-----')
    net = VGG('VGG19')
    
# 设置训练模式
ids=[int(i) for i in os.environ["CUDA_VISIBLE_DEVICE"].split(",")]
if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=ids)
    torch.backends.cudnn.benchmark = True
    
# 定义度量和优化
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# 训练
def tarin(epoch):
    train_loss = 0
    correct = 0
    total = 0
    # 训练阶段
    net.train()
    loop_train = tqdm(enumerate(train_loader), total =len(train_loader))
    for batch_idx,(inputs, targets)in loop_train:
        # 将数据转移到gpu上
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # 模型输出，更新梯度
        pred = net(inputs)
        loss = criterion(pred, targets)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()
        
        # 数据统计
        net.eval() # 关闭dropout
        train_loss += loss.item()  # .item() 取字典值
        _, predict = torch.max(pred.data, 1)  # 找出行最大值,一共有batchsize行
        total += targets.size(0)
        correct += predict.eq(targets.data).cpu().sum()
        loss = train_loss/(batch_idx+1)
        acc = 100.*correct/total
        # loss = compute_epoch_loss(net, train_loader)
        # acc = compute_accuracy(net, train_loader)
        loop_train.set_description(f'Epoch_train [{epoch}/{args.start_epoch+args.epoch}]')
        loop_train.set_postfix(loss = loss,acc = acc)
        # print('\n%d epoch done!'%epoch+1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return acc, loss

if __name__ =="__main__":
    best_acc = 0 
    for epoch in range(args.start_epoch, args.start_epoch+args.epoch):
        acc,loss = tarin(epoch)
        # 清除部分无用变量 
        torch.cuda.empty_cache()
        # 保存模型
        if acc > best_acc:
            # print('-----saving-----')
            cur_net = {
                'net': net,
                'acc': acc
            }
            if not os.path.exists('log'):
                os.mkdir('log')
            torch.save(cur_net, './log/epoch%d_acc%4f_loss%4f.pth'%(epoch+1,acc,loss))
            best_acc = acc