import torch
from datasets import PACS_all,PACS_singledomain,PACS_test,Domain_drop_all
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_dict = {'dog':0,'elephant':1,'giraffe':2,'guitar':3,'horse':4,'house':5,'person':6}
target_transform = lambda x:label_dict[x]

def get_domain_drop_dataloader(args):# 所有训练域的dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    Domain_drop = Domain_drop_all('PACS/train/',data_transforms,target_transform)
    Domain_drop_all_dataloader = DataLoader(Domain_drop,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True, worker_init_fn=worker_init_fn, generator=generator)
    return Domain_drop_all_dataloader

def get_dataloader_all(args):# 所有训练域的dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    PACS_all_dataset = PACS_all('PACS/train/',data_transforms,target_transform)
    PACS_all_dataloader = DataLoader(PACS_all_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True, worker_init_fn=worker_init_fn, generator=generator)
    return PACS_all_dataloader
    # return PACS_all_dataset
    
def get_dataloader_bydomain(args):#单个训练域的dataloader
    PACS_P_dataset = PACS_singledomain('PACS/train/photo',data_transforms,target_transform)
    PACS_A_dataset = PACS_singledomain('PACS/train/art_painting',data_transforms,target_transform)
    PACS_C_dataset = PACS_singledomain('PACS/train/cartoon',data_transforms,target_transform)

    PACS_P_dataloader = DataLoader(PACS_P_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True)
    PACS_A_dataloader = DataLoader(PACS_A_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True)
    PACS_C_dataloader = DataLoader(PACS_C_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True)
    
    return PACS_P_dataloader,PACS_A_dataloader ,PACS_C_dataloader 

def get_dataloader_test(args):# 测试集的dataloader
    test_dataset =  PACS_test('PACS/test',data_transforms)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=False)
    
    return test_dataloader

def setup_seed(seed):#设置种子
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    from config import get_parser
    # from utils import *
    import tqdm
    args = get_parser()
    device = 'mps'
    # setup_seed(0)
    # train_loader = get_dataloader_all(args)
    print(device)
    PACS_all_dataset = PACS_all('PACS/train/',data_transforms,target_transform)
    PACS_all_dataloader = DataLoader(PACS_all_dataset,batch_size=args.batchsize,num_workers=args.num_workers,shuffle=True)
    # train_loader = get_dataloader_all(args)
    # PACS_all_dataset = PACS_all('./PACS/train/')
    n_epochs = 2
    # print(PACS_all_dataset[1001])
    for data, target in PACS_all_dataloader:
        data.to(device)
        target.to(device)
        break


    # for a in PACS_all_dataloader:
    #     print(a)

    #     break
    # print(len(PACS_all_dataset))
    # print(PACS_all_dataset[0])
    # im = PACS_all_dataset[0][0]
    # im.show()