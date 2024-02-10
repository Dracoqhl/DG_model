import math
import random

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck
from .EFDMix import EFDMix
from .LayerDiscriminator import LayerDiscriminator
# from models.EFDMix import EFDMix

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, interval=75,network=None,
                 device=None,
                 domains=3,
                 domain_discriminator_flag=0,
                 grl=0,
                 lambd=0.,
                 drop_percent=0.33,
                 dropout_mode=0,
                 wrs_flag=0,
                 recover_flag=0,
                 layer_wise_flag=0,):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.percent = 0.5
        self.interval = interval
        self.efdmix = EFDMix(p=0.5, alpha=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 判断是否是18网络[2, 2, 2, 2],34网络
        if network == 18:
            layer_channels = [64, 64, 128, 256, 512]
        else:
            layer_channels = [64, 256, 512, 1024, 2048]

        self.device = device
        self.domain_discriminator_flag = domain_discriminator_flag
        self.drop_percent = drop_percent
        self.dropout_mode = dropout_mode

        self.recover_flag = recover_flag
        self.layer_wise_flag = layer_wise_flag

        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels=layer_channels[layer],
                num_classes=domains,
                grl=grl,
                reverse=True,
                lambd=lambd,
                wrs_flag=wrs_flag,
                )
            for i, layer in enumerate([0, 1, 2, 3, 4])])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def perform_dropout(self, feature, domain_labels, layer_index, layer_dropout_flag):
        domain_output = None
        if self.domain_discriminator_flag and self.training:
            index = layer_index
            percent = self.drop_percent
            domain_output, domain_mask = self.domain_discriminators[index](
                feature.clone(),
                domain_labels,
                percent=percent,
            )
            if self.recover_flag:
                domain_mask = domain_mask * domain_mask.numel() / domain_mask.sum()
            if layer_dropout_flag:
                feature = feature * domain_mask
        return feature, domain_output

    def forward(self, x, ground_truth=None, domain_labels=None, layer_drop_flag=None, epoch=None):
        # [bs*2, 3, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        domain_outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            x, domain_output = self.perform_dropout(x, domain_labels, layer_index=i + 1,
                                                    layer_dropout_flag=layer_drop_flag[i])
            if i == 0:
                x = self.efdmix(x)
            if domain_output is not None:
                domain_outputs.append(domain_output)

        # x = self.layer1(x)
        # x = self.efdmix(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # [bs*2, 512, 7, 7]

        # RSC Self-Challenge method on 7x7 feature map
        if self.training:
            if epoch % self.interval == 0:
                self.percent = 0.5 - (epoch / self.interval) * 0.2
            # forward with cloned feature map
            self.eval()
            x_new = x.clone().detach().requires_grad_()
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.fc(x_new_view)
            num_classes = output.shape[1]
            num_rois, num_channel, H, W = x_new.shape
            HW = H * W
            one_hot_sparse = F.one_hot(ground_truth, num_classes)
            one_hot = torch.sum(output * one_hot_sparse)

            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()  # [bs*2, 512, 7, 7]
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean  # [bs*2, 512]
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, dim=1)
            spatial_mean = spatial_mean.view(num_rois, HW)  # [bs*2, 49]
            self.zero_grad()

            if random.random() < 0.5:
                # ---------------------------- spatial -----------------------
                spatial_thres_percent = math.ceil(HW * 1 / 3.0)
                spatial_thres_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_thres_percent]
                spatial_thres_value = spatial_thres_value.view(num_rois, 1).expand(num_rois, 49)
                mask_all = torch.where(spatial_mean > spatial_thres_value,
                                       torch.zeros(spatial_mean.shape).cuda(),
                                       torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all.reshape(num_rois, H, W).view(num_rois, 1, H, W)
            else:
                # -------------------------- channel ----------------------------
                # TODO: 待测试 —— 论文用的都是 33.3%
                channel_thres_percent = math.ceil(num_channel * 1 / 3.2)
                channel_thres_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, channel_thres_percent]
                channel_thres_value = channel_thres_value.view(num_rois, 1).expand(num_rois, num_channel)
                mask_all = torch.where(channel_mean > channel_thres_value,
                                       torch.zeros(channel_mean.shape).cuda(),
                                       torch.ones(channel_mean.shape).cuda())
                mask_all = mask_all.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(num_rois, -1)
            x_new_view_after = self.fc(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            one_hot_sparse = F.one_hot(ground_truth, num_classes)
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0,
                                        change_vector,
                                        torch.zeros(change_vector.shape).cuda())
            batch_thres_percent = int(round(float(num_rois) * self.percent))
            batch_thres_value = torch.sort(change_vector, dim=0, descending=True)[0][batch_thres_percent]
            ignore_batch = change_vector.le(batch_thres_value)
            ignore_batch_idxs = ignore_batch.nonzero()[:, 0]
            mask_all[ignore_batch_idxs, :] = 1

            self.train()
            mask_all.requires_grad_()
            x = x * mask_all

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x,domain_outputs

class EFDMixResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(EFDMixResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.efdmix = EFDMix(p=0.5, alpha=0.1)
        self.percent = 0.5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ground_truth=None, epoch=None):
        # [bs*2, 3, 224, 224]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.efdmix(x)
        x = self.layer2(x)
        x = self.efdmix(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [bs*2, 512, 7, 7]

        # RSC Self-Challenge method on 7x7 feature map
        if False:
        # if self.training:
            interval = 75
            if epoch % interval == 0:
                self.percent = 0.5 - (epoch / interval) * 0.2
            # forward with cloned feature map
            self.eval()
            x_new = x.clone().detach().requires_grad_()
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.fc(x_new_view)
            num_classes = output.shape[1]
            num_rois, num_channel, H, W = x_new.shape
            HW = H * W
            one_hot_sparse = F.one_hot(ground_truth, num_classes)
            one_hot = torch.sum(output * one_hot_sparse)

            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()  # [bs*2, 512, 7, 7]
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean  # [bs*2, 512]
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, dim=1)
            spatial_mean = spatial_mean.view(num_rois, HW)  # [bs*2, 49]
            self.zero_grad()

            if random.random() < 0.5:
                # ---------------------------- spatial -----------------------
                spatial_thres_percent = math.ceil(HW * 1 / 3.0)
                spatial_thres_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_thres_percent]
                spatial_thres_value = spatial_thres_value.view(num_rois, 1).expand(num_rois, 49)
                mask_all = torch.where(spatial_mean > spatial_thres_value,
                                       torch.zeros(spatial_mean.shape).cuda(),
                                       torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all.reshape(num_rois, H, W).view(num_rois, 1, H, W)
            else:
                # -------------------------- channel ----------------------------
                # TODO: 待测试 —— 论文用的都是 33.3%
                channel_thres_percent = math.ceil(num_channel * 1 / 3.2)
                channel_thres_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, channel_thres_percent]
                channel_thres_value = channel_thres_value.view(num_rois, 1).expand(num_rois, num_channel)
                mask_all = torch.where(channel_mean > channel_thres_value,
                                       torch.zeros(channel_mean.shape).cuda(),
                                       torch.ones(channel_mean.shape).cuda())
                mask_all = mask_all.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(num_rois, -1)
            x_new_view_after = self.fc(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            one_hot_sparse = F.one_hot(ground_truth, num_classes)
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0,
                                        change_vector,
                                        torch.zeros(change_vector.shape).cuda())
            batch_thres_percent = int(round(float(num_rois) * self.percent))
            batch_thres_value = torch.sort(change_vector, dim=0, descending=True)[0][batch_thres_percent]
            ignore_batch = change_vector.le(batch_thres_value)
            ignore_batch_idxs = ignore_batch.nonzero()[:, 0]
            mask_all[ignore_batch_idxs, :] = 1

            self.train()
            mask_all.requires_grad_()
            x = x * mask_all

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def domain_resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def efdmix_resnet18(pretrained=True, **kwargs):
    model = EFDMixResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

