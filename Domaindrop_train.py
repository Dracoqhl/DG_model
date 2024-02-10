import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import *
# from torchvision.models import resnet34,resnet50,resnet101
from model.resnet_domain_drop import efdmix_resnet18,resnet18,resnet34,resnet50
from config import get_parser
import pandas as pd
from loss.KL_Loss import compute_kl_loss

def select_layers(layer_wise_prob):
        # layer_wise_prob: prob for layer-wise dropout
        layer_index = np.random.randint(len(args.discriminator_layers), size=1)[0]
        layer_select = args.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select - 1] = 1
        return layer_drop_flag

if __name__ == '__main__':
    
    args = get_parser()
    setup_seed(0)
    train_loader = get_domain_drop_dataloader(args)
    test_loader = get_single_dateloader(args)
    print("Dataset size: train %d, test %d" % (len(train_loader), len(test_loader)))
    n_class = 7
    # device = torch.device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # Model Definition
    if args.model == 18:
        model = resnet18(
            pretrained=False,
            num_classes=n_class,
            interval=0.75 * (args.epoch+args.begin_epoch),
            network=args.model,
            device=device,

            domains=3,
            domain_discriminator_flag=args.domain_discriminator_flag,
            grl=args.grl,
            lambd=args.lambd,
            drop_percent=args.drop_percent,
            wrs_flag=args.filter_WRS_flag,
            recover_flag=args.recover_flag, 
        )
    if args.model == 34:
        model = resnet34(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch),network=args.model)
    if args.model == 50:
        model = resnet50(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch),network=args.model,
            device=device,

            domains=3,
            domain_discriminator_flag=args.domain_discriminator_flag,
            grl=args.grl,
            lambd=args.lambd,
            drop_percent=args.drop_percent,
            wrs_flag=args.filter_WRS_flag,
            recover_flag=args.recover_flag,        
            )
    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    # LOAD = False
    # Train = False
    score = 0
    # 损失函数和优化器定义
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), weight_decay=.01, momentum=.9, nesterov=False, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=int(len(train_loader)), epochs=(args.epoch+args.begin_epoch), pct_start=0.1)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor = 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    

    if args.train:
        # begin_epoch = 0
        if args.isLoad:
            # 如果需要导入现有的模型
            state_dict = torch.load('Checkpoints/resnet34_test_400.pth')
            model.load_state_dict(state_dict)
            # begin_epoch = 200
        
        model = model.to(device)

        # layer_wise_prob = args.layer_wise_prob
        domain_criterion = nn.CrossEntropyLoss()

        domain_discriminator_flag = args.domain_discriminator_flag
        domain_loss_flag = args.domain_loss_flag
        discriminator_layers = args.discriminator_layers
        layer_wise_prob = args.layer_wise_prob
        for epoch in range(args.begin_epoch+1, args.begin_epoch+args.epoch+1):
            CE_loss = 0.0
            batch_num = 0.0
            class_right = 0.0
            class_total = 0.0

            CE_domain_loss = [0.0 for i in range(5)]
            domain_right = [0.0 for i in range(5)]
            CE_domain_losses_avg = 0.0
            KL_loss = 0.0


            train_loss = 0.0
            right_sample = 0
            total_sample = 0
            test_right_sample = 0
            test_total_sample = 0
            # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            model.train()
            for data, target,domain_l in tqdm(train_loader):
                
                data = data.to(device)
                target = target.to(device) 
                domain_l = domain_l.to(device)

                layer_drop_flag = select_layers(layer_wise_prob= layer_wise_prob)
                # print(f'layer:{layer_drop_flag}')

                data_flip = torch.flip(data, (3,)).detach().clone()
                data = torch.cat((data, data_flip))
                target = torch.cat((target, target))
                domain_l = torch.cat((domain_l, domain_l))

                ''' '''
            # def forward(self, x, ground_truth=None, domain_labels=None, layer_drop_flag=None, epoch=None):
                (class_logit, domain_logit) = model(x=data, domain_labels=domain_l, ground_truth = target, layer_drop_flag=layer_drop_flag, epoch = epoch)
                class_loss = criterion(class_logit, target)
                CE_loss += class_loss
                domain_losses_avg = torch.tensor(0.0).to(device=device)

                if domain_discriminator_flag == 1:
                    domain_losses = []
                    for i, logit in enumerate(domain_logit):
                        domain_loss = domain_criterion(logit, domain_l)
                        domain_losses.append(domain_loss)
                        CE_domain_loss[i] += domain_loss
                    domain_losses = torch.stack(domain_losses, dim=0)
                    domain_losses_avg = domain_losses.mean(dim=0)
                CE_domain_losses_avg += domain_losses_avg

                loss = 0.0
                loss += class_loss
                if domain_loss_flag == 1:
                    loss += domain_losses_avg
                if args.KL_Loss == 1:
                    batch_size = int(class_logit.shape[0] / 2)
                    class_logit_1 = class_logit[:batch_size]
                    class_logit_2 = class_logit[batch_size:]
                    kl_loss = compute_kl_loss(class_logit_1, class_logit_2, T=args.KL_Loss_T)
                    loss += args.KL_Loss_weight * kl_loss
                    KL_loss += kl_loss

                optimizer.zero_grad()
                # for opt in optimizer:
                #     opt.zero_grad()
                loss.backward()
                optimizer.step()
                # for opt in optimizer:
                #     opt.step()

                _, class_pred = class_logit.max(dim=1)
                class_right_batch = torch.sum(class_pred == target.data)
                class_right += class_right_batch

                domain_right_batch = [torch.tensor(0.0).cuda() for i in range(5)]
                if domain_discriminator_flag == 1:
                    for i, logit in enumerate(domain_logit):
                        _, domain_pred = logit.max(dim=1)
                        domain_right_batch[i] = torch.sum(domain_pred == domain_l.data)
                        domain_right[i] += domain_right_batch[i]
                batch_num += 1

                data_shape = data.shape[0]
                class_total += data_shape

                CE_loss = float(CE_loss) / batch_num
                CE_domain_losses_avg = float(CE_domain_losses_avg / batch_num)
                CE_domain_loss = [float(loss / batch_num) for loss in CE_domain_loss]

                class_acc = float(class_right) / class_total
                domain_acc = [float(right / class_total) for right in domain_right]

                ''' '''
                # optimizer.zero_grad()
                # output, domain_logit = model(x=data, domain_labels=domain_l, layer_drop_flag=layer_drop_flag).to(device)
                # # output = model(data, target, epoch).to(device)
                # # output = model(data)
                # pred = torch.argmax(output,dim=1)

                # right_sample += torch.sum(pred==target)
                # total_sample += target.shape[0]

                # loss = criterion(output, target)
                # loss.backward()
                # optimizer.step()
                
                # train_loss += loss.item()*data.size(0)
                scheduler.step()

            with torch.no_grad():
                model.eval()
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device) 
                    output,_ = model(x=data, domain_labels=domain_l, ground_truth = target, layer_drop_flag=layer_drop_flag, epoch = epoch)
                    # output = model(data, target, '000',epoch).to(device)
                    pred = torch.argmax(output,dim=1)

                    test_right_sample += torch.sum(pred==target)
                    test_total_sample += target.shape[0]

            train_loss = train_loss/len(train_loader.sampler)
            test_acc = test_right_sample/test_total_sample
            if test_acc > score and test_acc > 0.7:
                score = test_acc
                torch.save(model.state_dict(), f'Checkpoints/domain_resnet{args.model}_test_{epoch}.pth')
            print('Epoch: {} \tTraining Loss: {:.6f}\tTraing Acc: {:6f}\tTesting Acc: {:6f}'.format(epoch, CE_loss, class_acc, test_acc))
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
