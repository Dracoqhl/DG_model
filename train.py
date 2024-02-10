import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import *
# from torchvision.models import resnet34,resnet50,resnet101
from model.resnet import resnet18,resnet34,resnet50
from config import get_parser
import pandas as pd



if __name__ == '__main__':
    
    args = get_parser()
    setup_seed(0)
    train_loader = get_dataloader_all(args)
    test_loader = get_dataloader_test(args)
    print("Dataset size: train %d, test %d" % (len(train_loader), len(test_loader)))
    n_class = 7
    # device = torch.device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model Definition
    if args.model == 18:
        model = resnet18(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
    if args.model == 34:
        model = resnet34(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
    if args.model == 50:
        model = resnet50(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
    score = 0
    # 损失函数和优化器定义
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), weight_decay=.01, momentum=.9, nesterov=False, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=int(len(train_loader)), epochs=(args.epoch+args.begin_epoch), pct_start=0.1)

    if args.train:
        if args.isLoad:
            # 如果需要导入现有的模型
            state_dict = torch.load('Checkpoints/resnet34_test_400.pth')
            model.load_state_dict(state_dict)
        
        model = model.to(device)

        for epoch in range(args.begin_epoch+1, args.begin_epoch+args.epoch+1):
            train_loss = 0.0
            right_sample = 0
            total_sample = 0
            # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model.train()
            for data, target in tqdm(train_loader):
                
                data = data.to(device)
                target = target.to(device) 

                data_flip = torch.flip(data, (3,)).detach().clone()
                data = torch.cat((data, data_flip))
                target = torch.cat((target, target))

                optimizer.zero_grad()
                output = model(data, target, epoch).to(device)
                # output = model(data)
                pred = torch.argmax(output,dim=1)

                right_sample += torch.sum(pred==target)
                total_sample += target.shape[0]

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()*data.size(0)
                scheduler.step()

            train_loss = train_loss/len(train_loader.sampler)
            torch.save(model.state_dict(), f'Checkpoints/resnet{args.model}_test_{epoch}.pth')
            print('Epoch: {} \tTraining Loss: {:.6f}\tTraing Acc: {:6f}'.format(epoch, train_loss, right_sample/total_sample))
    else:
        ID = []
        LABEL = []
        state_dict = torch.load('Checkpoints/resnet34_test_250.pth')
        
        model.load_state_dict(state_dict)
        # model = model.module
        
        model = model.to(device)
        model.eval()
        test_loader = get_dataloader_test(args)
        label_reverse_list = ['dog','elephant','giraffe','guitar','horse','house','person']
        with torch.no_grad():
            for index,data in test_loader:
                data = data.to(device)
                output = model(data,device)
                pred = torch.argmax(output,dim=1).cpu().numpy().tolist()
                ID += torch.tensor(index).numpy().tolist()
                LABEL += pred
        for i in range(len(LABEL)):
            LABEL[i] = label_reverse_list[LABEL[i]]

        res = pd.DataFrame([ID,LABEL]).T
        res.columns = ['ID','label']
        res.to_csv('result.csv',index=False)

        print('test finish!\tpredict file save in result.csv')
