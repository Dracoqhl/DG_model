import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import *
from model.resnet_domain_drop import domain_resnet18
from model.resnet import resnet18,resnet34,resnet50
# from model import ResNet18,ResNet34
import pandas as pd
import warnings
from config import get_parser
warnings.filterwarnings("ignore")

from datasets import *


if __name__ == '__main__':

    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # label_reverse_list = ['dog','elephant','giraffe','guitar','horse','house','person']

    model_dic = {'18':'resnet18_test_355.pth',
                 '34':'resnet34_test_385.pth',
                 '50':'resnet50_test_413.pth',
                 'dom_18':'resnet18_test_386.pth'
                 }
    setup_seed(0)
    n_class = 7
    model_list = []
    for i in model_dic:
        if i == '18' and model_dic[i] != '':
            model = resnet18(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
        elif i == '34' and model_dic[i] != '':
            model = resnet34(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
        elif i == '50' and model_dic[i] != '':
            model = resnet50(pretrained=False,num_classes=n_class,interval=0.75 * (args.epoch+args.begin_epoch))
        else:
            continue
        state_dict = torch.load(f'Checkpoints/{model_dic[i]}')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        model_list.append(model)
    
    ID = []
    LABEL = []
    test_loader = get_dataloader_test(args)
    label_reverse_list = ['dog','elephant','giraffe','guitar','horse','house','person']
    # 把新模型加进去
    domain_resnet18 = domain_resnet18(
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
    state_dict = torch.load(f'Checkpoints/resnet18_test_386.pth')
    domain_resnet18.load_state_dict(state_dict)
    domain_resnet18 = domain_resnet18.to(device)
    domain_resnet18.eval()
    with torch.no_grad():
        # [0.1, 0.4,0.3,0.2]
        weights = [0.1, 0.4,0.3,0.2]
        for index,data in test_loader:
            data = data.to(device)
            outputs = []
            for idx,j in enumerate(model_list):
                outputs.append(j(data, device))
        

            output,_ = domain_resnet18(x=data, domain_labels='0', ground_truth = '0', layer_drop_flag=[0,0,0,0], epoch = 0)
            outputs.append(output)
            outputs = torch.stack(outputs)  # 堆叠所有模型的输出

            # 计算加权平均输出
            weighted_sum = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                weighted_sum += weights[i] * output
            weighted_avg_output = weighted_sum / sum(weights)

            # avg_output = torch.mean(outputs, dim=0)
            pred = torch.argmax(weighted_avg_output,dim=1).cpu().numpy().tolist()
            ID += torch.tensor(index).numpy().tolist()
            LABEL += pred
        for i in range(len(LABEL)):
            LABEL[i] = label_reverse_list[LABEL[i]]

        res = pd.DataFrame([ID,LABEL]).T
        res.columns = ['ID','label']
        res.to_csv('result.csv',index=False)

    print('test finish!\tpredict file save in result.csv')
