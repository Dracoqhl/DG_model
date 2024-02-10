import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os
dom_dic = {'art_painting':0, 'cartoon':1, 'photo':2}
class Domain_drop_all(Dataset): #包含所有训练域的数据集
    def __init__(self, root_dir, transform=None,target_transform=None):
        self.root_dir = root_dir 
        print(self.root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.domain = os.listdir(self.root_dir)
        for i in self.domain:
            # 如果以'.'开头，则删除
            if i.startswith('.'):
                self.domain.remove(i)
        self.label = os.listdir(os.path.join(self.root_dir,self.domain[0]))
        for i in self.label:
            if i.startswith('.'):
                self.label.remove(i)
        print(f'labels:{self.label}')
        print(f'domains:{self.domain}')
        print(f'root_dir:{self.root_dir}')
        self.labels = []
        self.domains = []
        self.images = []
        for domain in self.domain:
            for label in self.label:
                for image in os.listdir(os.path.join(self.root_dir, domain, label)):
                    self.labels.append(label)
                    self.images.append(os.path.join(domain, label, image))
                    self.domains.append(dom_dic[domain])
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = self.labels[index]
        domain = self.domains[index]
        
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        if self.target_transform:
            label = self.target_transform(label)
            # domain = self.target_transform(domain)
        return sample,label,domain
    
class PACS_all(Dataset): #包含所有训练域的数据集
    def __init__(self, root_dir, transform=None,target_transform=None):
        self.root_dir = root_dir 
        print(self.root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.domain = os.listdir(self.root_dir)
        for i in self.domain:
            # 如果以'.'开头，则删除
            if i.startswith('.'):
                self.domain.remove(i)
        self.label = os.listdir(os.path.join(self.root_dir,self.domain[0]))
        for i in self.label:
            if i.startswith('.'):
                self.label.remove(i)
        print(f'labels:{self.label}')
        print(f'domains:{self.domain}')
        print(f'root_dir:{self.root_dir}')
        self.labels = []
        self.images = []
        for domain in self.domain:
            for label in self.label:
                for image in os.listdir(os.path.join(self.root_dir, domain, label)):
                    self.labels.append(label)
                    self.images.append(os.path.join(domain, label, image))
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        if self.target_transform:
            label = self.target_transform(label)
        return sample,label

class PACS_singledomain(Dataset):#训练集中单个域的数据集
    
    def __init__(self, root_dir, transform=None,target_transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.target_transform = target_transform
        self.label = os.listdir(self.root_dir)
        for i in self.label:
            if i.startswith('.'):
                self.label.remove(i)
        print(f'labels:{self.label}')
        print(f'root_dir:{self.root_dir}')
        self.labels = []
        self.images = []
        for label in self.label:
            for image in os.listdir(os.path.join(self.root_dir, label)):
                self.labels.append(label)
                self.images.append(os.path.join(label, image))
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return sample,label
    
class PACS_test(Dataset):#测试数据集
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.images = os.listdir(self.root_dir)
        if '.ipynb_checkpoints' in self.images:
            pop_index = self.images.index('.ipynb_checkpoints')
            self.images.pop(pop_index)
        self.images = sorted(self.images,key=lambda x:int(x.split('.')[0]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = Image.open(img_path)
        
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return index,sample
    
if __name__ == '__main__':
    from config import get_parser
    # from utils import *
    import tqdm
    args = get_parser()
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    # setup_seed(0)
    # train_loader = get_dataloader_all(args)
    PACS_all_dataset = PACS_all('./PACS/train/')
    n_epochs = 2
    print(len(PACS_all_dataset))
    print(PACS_all_dataset[0])
    im = PACS_all_dataset[0][0]
    im.show()
    