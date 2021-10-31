import time
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

from vgg import VGG

from utils import (cvtColor, resize_image, preprocess_input)

from torchvision import transforms
data_transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == "__main__":
    vgg = VGG("VGG19")
    # load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weight_path = "/home/zhaiyize/models/vgg/log/epoch26_acc88.000000_loss0.015849.pth"  # 导入权重参数
    state = torch.load(model_weight_path)
    vgg = state['net']
    vgg = vgg.to(device)
    vgg.eval() 

    # class
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    dir_origin_path = "/home/zhaiyize/testimg"
    dir_save_path   = "/home/zhaiyize/vgg/result"  

    for file in os.listdir(dir_origin_path):
        if not file.endswith(".jpg" or ".png"):
            continue 
        img = Image.open(dir_origin_path +os.sep+ file)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).float() # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]
        img = img.cuda()
        # print(img.shape)
        with torch.no_grad():
            # predict class
            output = vgg(img)
            prob = torch.softmax(output, dim=0)  # 经过softmax函数将输出变为概率分布
            # print(prob)
            value, predicted = torch.max(output.data, 1)
            print(predicted.item())
            # print(value)
            pred_class = classes[predicted.item()]
            print(pred_class)