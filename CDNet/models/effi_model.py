import math
import os

import torchvision
from typing import Optional, Union, List

import numpy as np
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder

from segmentation_models_pytorch.unet.decoder import UnetDecoder
import torch.nn.functional as F
import torchsnooper

from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from torch.distributions.beta import Beta

import torchvision.transforms.functional as trans_F

import timm
from timm.models.layers import trunc_normal_


class Weakly_Supervised_Lesion_Localization(nn.Module):
    def __init__(self, att_in=32, pool='GAP'):
        super(Weakly_Supervised_Lesion_Localization, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.conv1d = nn.Conv1d(in_channels=att_in, out_channels=1, kernel_size=(1, 1))
        self.apply(self._init_weights)

    def forward(self, feat7, attention_MS):
        B, C, H, W = feat7.size()
        _, M, AH, AW = attention_MS.size()

        # match size
        if AH != H or AW != W:
            attention_MS = F.upsample_bilinear(attention_MS, size=(H, W))

        # MCSP

        # feature_matrix: (B, M, C) -> (B, M * C)
        feature_matrix = (torch.einsum('imjk,injk->imn', (attention_MS, feat7)) / float(H * W))  # B * 32 * 2048
        feature_matrix = self.conv1d(feature_matrix).view(B, C)

        return feature_matrix

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


class Multi_Stage_Attention_Module(nn.Module):

    def __init__(self, config):
        super(Multi_Stage_Attention_Module, self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        self.attention_texture = nn.Sequential(
            nn.Conv2d(1024, config.attention_map_num, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(config.attention_map_num, eps=0.001),
            nn.ReLU(inplace=True))
        self.attention_semantic = nn.Sequential(
            nn.Conv2d(2048, config.attention_map_num, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(config.attention_map_num, eps=0.001),
            nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.apply(self._init_weights)

    def forward(self, x2, x1):
        # print(x.size())
        semantic_map = self.attention_semantic(x1)  # 32 channels, size
        texture_map = self.attention_texture(x2)
        attention_output = semantic_map + self.avgpool(texture_map)
        return attention_output

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class effiEncoder(nn.Module):
    def __init__(self, config, model_idx="Eff_b3", input_size=224):
        super(effiEncoder, self).__init__()
        # load backbone and optimize its architecture
        if model_idx == "Eff_b0":
            timm_idx = "tf_efficientnet_b0"
            self.feat_channel_list = [16, 24, 40, 112, 320]
        elif model_idx == "Eff_b3":
            timm_idx = "tf_efficientnet_b3"
            self.feat_channel_list = [24, 32, 48, 136, 384]
        else:
            raise NotImplementedError(f"Not implemented {model_idx}")
        self.effinet_encoder = timm.create_model(timm_idx, pretrained=True, features_only=True)

        if input_size == 224:
            self.feat_size_list = [112, 56, 28, 14, 7]
            self.feature_matrix_dim = 2048
        elif input_size == 384:
            self.feat_size_list = [192, 96, 48, 24, 12]
            self.feature_matrix_dim = None
        else:
            raise NotImplementedError(f"Not implemented {input_size}")

        assert self.feature_matrix_dim is not None, "Please set the feature_matrix_dim"

        # 适应efficient net中间层输出的通道数
        self.adap_conv14 = nn.Conv2d(in_channels=self.feat_channel_list[-2],
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1))
        self.adap_conv7 = nn.Conv2d(in_channels=self.feat_channel_list[-1],
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))

        self.cls_Weakly_Supervised_Lesion_Localization = Weakly_Supervised_Lesion_Localization(pool='GAP')
        self.attention_module_cls = Multi_Stage_Attention_Module(config)

        self.M = config.attention_map_num
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

        self.class_num = config.class_num
        self.cls_head = self._get_cls_head(in_feat=self.feature_matrix_dim, num_classes=config.class_num)

    def _get_cls_head(self, in_feat=2048, num_classes=5):
        return nn.Sequential(
            nn.LeakyReLU(), nn.Linear(in_features=in_feat, out_features=num_classes))

    def forward(self, x, return_cls_feat=False, blocking=False):
        # 2&1atten
        _, _, _, feat14, feat7 = self.effinet_encoder(x)
        feat14 = self.adap_conv14(feat14)  # bs * 1024 * 14 * 14
        feat7 = self.adap_conv7(feat7)  # # bs * 2048 * 7 * 7

        attention_maps_cls = self.attention_module_cls(feat14, feat7)  # 2atten
        feature_matrix_cls = self.cls_Weakly_Supervised_Lesion_Localization(feat7, attention_maps_cls)

        # feature_matrix_cls = torch.sign(feature_matrix_cls) * torch.sqrt(torch.abs(feature_matrix_cls))
        # feature_matrix_cls = F.normalize(feature_matrix_cls, dim=-1)

        if blocking:
            cls_pred = self.cls_head(feature_matrix_cls.detach())
        else:
            cls_pred = self.cls_head(feature_matrix_cls)
        # cls_pred:bs * class_num(5), att_maps_cls:bs * 32 * 7 * 7
        if return_cls_feat:
            return cls_pred, attention_maps_cls, feature_matrix_cls
        else:
            return cls_pred, attention_maps_cls


class effiEncoder_ZoomIn(nn.Module):
    def __init__(self,
                 config,
                 model_idx: str = "Eff_b3",
                 cls_num: int = 5,
                 input_size: int = 224):
        super(effiEncoder_ZoomIn, self).__init__()
        self.eff_back = effiEncoder(config=config, model_idx=model_idx, input_size=input_size)
        self.en_fc = nn.Sequential(nn.LeakyReLU(), nn.Linear(2 * 2048, cls_num))

    @torch.no_grad()
    def gen_Zoom_in(self, images, atten_maps, return_crop=True, att_thres_theta=0.5, padding_ratio=0.1):
        batches, _, imgH, imgW = images.size()
        crop_images = []
        height_min_ids, height_max_ids = [], []
        width_min_ids, width_max_ids = [], []

        for batch_index in range(batches):
            with torchsnooper.snoop():
                atten_map = atten_maps[batch_index:batch_index + 1]
                theta_c = att_thres_theta * atten_map.max()

                try:
                    crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c

                    nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                    H_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)  # Top
                    H_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH - 1)  # Bottom
                    W_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)  # Left
                    W_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW - 1)  # Right

                except Exception as e:
                    print(e)

                    crop_mask[0, 0, 112, 112] = 1.

                    nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                    H_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)  # Top
                    H_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH - 1)  # Bottom
                    W_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)  # Left
                    W_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW - 1)  # Right

                if return_crop:
                    crop_images.append(
                        F.interpolate(
                            images[batch_index: batch_index + 1, :,
                            H_min:H_max, W_min:W_max],
                            size=(imgH, imgW), mode='bilinear')
                    )

                height_min_ids.append(H_min), height_max_ids.append(H_max)
                width_min_ids.append(W_min), width_max_ids.append(W_max)

        if return_crop:
            crop_images = torch.cat(crop_images, dim=0)

        return crop_images, height_min_ids, height_max_ids, width_min_ids, width_max_ids

    def forward(self, x, zoom_in=True):
        cls_pred, attention_maps_cls, feature_matrix_cls = self.eff_back(x, return_cls_feat=True)
        if zoom_in:
            atten_maps = torch.mean(
                attention_maps_cls.detach().clone(), dim=1, keepdim=True)
            crop_images, _, _, _, _, = self.gen_Zoom_in(
                images=x.detach().clone(), atten_maps=atten_maps)
            cls_pred2, attention_maps_cls2, feature_matrix_cls2 = \
                self.eff_back(crop_images, return_cls_feat=True)
            en_pred = self.en_fc(
                torch.cat([feature_matrix_cls, feature_matrix_cls2], dim=1)
            )
            return cls_pred, attention_maps_cls, feature_matrix_cls, cls_pred2, attention_maps_cls2, feature_matrix_cls2, en_pred

        else:
            return cls_pred, attention_maps_cls, feature_matrix_cls


class effiEncoder_MSFE(nn.Module):
    def __init__(self, config, model_idx="Eff_b3", input_size=224):
        super(effiEncoder_MSFE, self).__init__()
        # load backbone and optimize its architecture
        if model_idx == "Eff_b0":
            timm_idx = "tf_efficientnet_b0"
            self.feat_channel_list = [16, 24, 40, 112, 320]
        elif model_idx == "Eff_b3":
            timm_idx = "tf_efficientnet_b3"
            self.feat_channel_list = [24, 32, 48, 136, 384]
        else:
            raise NotImplementedError(f"Not implemented {model_idx}")
        self.effinet_encoder = timm.create_model(timm_idx, pretrained=True, features_only=True)

        if input_size == 224:
            self.feat_size_list = [112, 56, 28, 14, 7]
            self.feature_matrix_dim = 2048
        elif input_size == 384:
            self.feat_size_list = [192, 96, 48, 24, 12]
            self.feature_matrix_dim = None
        else:
            raise NotImplementedError(f"Not implemented {input_size}")

        assert self.feature_matrix_dim is not None, "Please set the feature_matrix_dim"

        self.attention_module_cls = Multi_Stage_Attention_Module(config)

        self.M = config.attention_map_num

        self.adap_conv14 = nn.Conv2d(in_channels=self.feat_channel_list[-2],
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1))
        self.adap_conv7 = nn.Conv2d(in_channels=self.feat_channel_list[-1],
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))

        self.cls_Weakly_Supervised_Lesion_Localization = Weakly_Supervised_Lesion_Localization(pool='GAP')

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x, return_cls_feat=False):

        _, _, _, feat14, feat7 = self.effinet_encoder(x)

        feat14 = self.adap_conv14(feat14)  # bs * 1024 * 14 * 14
        feat7 = self.adap_conv7(feat7)  # # bs * 2048 * 7 * 7

        attention_maps_cls = self.attention_module_cls(feat14, feat7)  # 2atten

        feature_matrix_cls = self.cls_Weakly_Supervised_Lesion_Localization(feat7, attention_maps_cls)

        # feature_matrix_cls = torch.sign(feature_matrix_cls) * torch.sqrt(torch.abs(feature_matrix_cls))
        # feature_matrix_cls = F.normalize(feature_matrix_cls, dim=-1)

        # cls_pred:bs * class_num(5), att_maps_cls:bs * 32 * 7 * 7

        if return_cls_feat:
            return attention_maps_cls, feature_matrix_cls
        else:
            return attention_maps_cls


class CDNet_Effib3(nn.Module):
    def __init__(self,
                 config,
                 model_idx: str = "Eff_b3",
                 cls_num: int = 5,
                 input_size: int = 224,
                 con_proj_head: str = "mlp",
                 con_feat_dim: int = 128,
                 bbox_threshold: float = 0.6):
        super(CDNet_Effib3, self).__init__()
        self.eff_back = effiEncoder_MSFE(config=config, model_idx=model_idx, input_size=input_size)

        self.in_size = input_size
        self.effi_dim = self.eff_back.feature_matrix_dim
        self.con_proj_head = con_proj_head
        self.con_feat_dim = con_feat_dim
        self.bbox_threshold = bbox_threshold

        self.ccrop_area = input_size ** 2
        self.ccrop_scale = [0.3, 0.5]
        self.ccrop_ratio = [0.6, 1.4]
        self.ccrop_log_ratio = torch.log(torch.tensor(self.ccrop_ratio))
        self.ccrop_beta = Beta(0.1, 0.1)

        self.cls_head = nn.Sequential(nn.LeakyReLU(), nn.Linear(in_features=self.effi_dim, out_features=cls_num))
        self.en_fc = nn.Sequential(nn.LeakyReLU(), nn.Linear(2 * self.effi_dim, cls_num))

        if con_proj_head == 'linear':
            self.con_proj_head = nn.Linear(self.effi_dim, self.con_feat_dim)

        elif con_proj_head == 'mlp':
            self.con_proj_head = nn.Sequential(
                nn.Linear(self.effi_dim, self.con_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.con_feat_dim, self.con_feat_dim)
            )

        else:
            raise NotImplementedError('head not supported: {}'.format(con_proj_head))

    @torch.no_grad()
    def gen_Zoom_in(self, images, atten_maps, return_crop=True, att_thres_theta=0.5, padding_ratio=0.1):
        """

        :param images:
        :param atten_maps:
        :param return_crop:
        :param att_thres_theta:
        :param padding_ratio:
        :return:
        """
        batches, _, imgH, imgW = images.size()
        crop_images = []
        height_min_ids, height_max_ids = [], []
        width_min_ids, width_max_ids = [], []

        for batch_index in range(batches):
            with torchsnooper.snoop():
                atten_map = atten_maps[batch_index:batch_index + 1]
                theta_c = att_thres_theta * atten_map.max()

                try:
                    crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c

                    nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                    H_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)  # Top
                    H_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH - 1)  # Bottom
                    W_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)  # Left
                    W_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW - 1)  # Right

                except Exception as e:
                    print(e)

                    crop_mask[0, 0, 112, 112] = 1.

                    nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                    H_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)  # Top
                    H_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH - 1)  # Bottom
                    W_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)  # Left
                    W_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW - 1)  # Right

                if return_crop:
                    crop_images.append(
                        F.interpolate(
                            images[batch_index: batch_index + 1, :,
                            H_min:H_max, W_min:W_max],
                            size=(imgH, imgW), mode='bilinear')
                    )

                height_min_ids.append(H_min), height_max_ids.append(H_max)
                width_min_ids.append(W_min), width_max_ids.append(W_max)

        if return_crop:
            crop_images = torch.cat(crop_images, dim=0)

        return crop_images, height_min_ids, height_max_ids, width_min_ids, width_max_ids

    @torch.no_grad()
    def gen_BBox(self, atten_maps):
        """

        :param atten_maps:
        :return:
        """
        bs = atten_maps.size(0)
        bboxs = []
        for idx in range(bs):
            att_map = atten_maps[idx:idx + 1]
            theta_c = self.bbox_threshold * att_map.max()

            crop_mask = F.interpolate(att_map, size=(self.in_size, self.in_size), mode='bilinear') >= theta_c
            try:
                # crop_mask = crop_mask.to("cpu")
                crop_mask[:, :, int(self.in_size / 2), int(self.in_size / 2)] = 1
                nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                height_min = max(int(nonzero_indices[:, 0].min().item()), 0) / self.in_size
                height_max = min(int(nonzero_indices[:, 0].max().item()), self.in_size) / self.in_size
                width_min = max(int(nonzero_indices[:, 1].min().item()), 0) / self.in_size
                width_max = min(int(nonzero_indices[:, 1].max().item()), self.in_size) / self.in_size

            except Exception as e:
                print(e)

                crop_mask[0, 0, 112, 112] = 1.

                nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
                height_min = max(int(nonzero_indices[:, 0].min().item()), 0)  # Top
                height_max = min(int(nonzero_indices[:, 0].max().item()), self.in_size - 1)  # Bottom
                width_min = max(int(nonzero_indices[:, 1].min().item()), 0)  # Left
                width_max = min(int(nonzero_indices[:, 1].max().item()), self.in_size - 1)  # Right

            box = [height_min, width_min, height_max, width_max]
            bboxs.append(box)
        return bboxs

    @torch.no_grad()
    def contrastive_crop(self, img, box):
        """

        :param img:
        :param box:
        :return:
        """
        for _ in range(10):
            target_area = self.ccrop_area * torch.empty(1).uniform_(self.ccrop_scale[0], self.ccrop_scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(self.ccrop_log_ratio[0], self.ccrop_log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= self.in_size and 0 < h <= self.in_size:
                h0, w0, h1, w1 = box
                ch0 = max(int(self.in_size * h0) - h // 2, 0)
                ch1 = min(int(self.in_size * h1) - h // 2, self.in_size - h)
                cw0 = max(int(self.in_size * w0) - w // 2, 0)
                cw1 = min(int(self.in_size * w1) - w // 2, self.in_size - w)

                i = ch0 + int((ch1 - ch0) * self.ccrop_beta.sample())
                j = cw0 + int((cw1 - cw0) * self.ccrop_beta.sample())

                break
        else:
            # Fallback to central crop
            in_ratio = float(self.in_size) / float(self.in_size)
            if in_ratio < min(self.ccrop_ratio):
                w = self.in_size
                h = int(round(w / min(self.ccrop_ratio)))
            elif in_ratio > max(self.ccrop_ratio):
                h = self.in_size
                w = int(round(h * max(self.ccrop_ratio)))
            else:  # whole image
                w = self.in_size
                h = self.in_size
            i = (self.in_size - h) // 2
            j = (self.in_size - w) // 2

        cimg = trans_F.resized_crop(img, i, j, h, w, size=[self.in_size, self.in_size])
        return cimg, i, j, h, w

    @torch.no_grad()
    def contrastive_multi_zoom_strategy(self, imgs, atten_maps):
        """

        :param imgs:
        :param atten_maps:
        :return:
        """
        bs = imgs.size(0)
        ccroped_img_1 = []
        ccroped_img_2 = []

        cc_box1, cc_box2 = [], []
        bboxs = self.gen_BBox(atten_maps=atten_maps)

        for (idx, box) in zip(range(bs), bboxs):
            img = imgs[idx:idx + 1]
            cimg, i1, j1, h1, w1 = self.contrastive_crop(img=img, box=box)
            ccroped_img_1.append(cimg)
            cc_box1.append((i1, j1, h1, w1))

            cimg2, i2, j2, h2, w2 = self.contrastive_crop(img=img, box=box)
            ccroped_img_2.append(cimg2)
            cc_box2.append((i2, j2, h2, w2))

        ccroped_img_1 = torch.cat(ccroped_img_1, dim=0)
        ccroped_img_2 = torch.cat(ccroped_img_2, dim=0)
        cc_box1 = torch.tensor(cc_box1)
        cc_box2 = torch.tensor(cc_box2)

        return ccroped_img_1, ccroped_img_2, cc_box1, cc_box2

    def forward(self, x, blocking=True, con_proj=True, ccrop=True, phase="train", return_ccrop_img=False,
                return_zoomin_img=False):
        """
        forward
        :param x:
        :param blocking:
        :param con_proj:
        :param ccrop:
        :param phase:
        :param return_ccrop_img:
        :param return_zoomin_img:
        :return:
        """

        att_maps_cls, feat_mat_cls = self.eff_back(x, return_cls_feat=True)

        atten_maps = torch.mean(att_maps_cls.detach(), dim=1, keepdim=True)

        if phase == "train" and ccrop:
            cc_img_1, cc_img_2, cc_box1, cc_box2 = self.contrastive_multi_zoom_strategy(imgs=x, atten_maps=atten_maps)

            # Local Branch
            atten_maps_cc1, feat_mat_cls_cc1 = self.eff_back(cc_img_1, return_cls_feat=True)
            atten_maps_cc2, feat_mat_cls_cc2 = self.eff_back(cc_img_2, return_cls_feat=True)

            feat_mat_cls_cc_en = feat_mat_cls_cc1 + feat_mat_cls_cc2

            if blocking:
                feat_mat_cls_ = feat_mat_cls.detach()
                feat_mat_cls_cc1_ = feat_mat_cls_cc1.detach()
                feat_mat_cls_cc2_ = feat_mat_cls_cc2.detach()
                feat_mat_cls_cc_en_ = feat_mat_cls_cc_en.detach()
            else:
                feat_mat_cls_ = feat_mat_cls
                feat_mat_cls_cc1_ = feat_mat_cls_cc1
                feat_mat_cls_cc2_ = feat_mat_cls_cc2
                feat_mat_cls_cc_en_ = feat_mat_cls_cc_en

            cls_pred = self.cls_head(feat_mat_cls_)
            cls_pred_cc1 = self.cls_head(feat_mat_cls_cc1_)
            cls_pred_cc2 = self.cls_head(feat_mat_cls_cc2_)
            en_pred = self.en_fc(torch.cat([feat_mat_cls_, feat_mat_cls_cc_en_], dim=1))

            if con_proj:
                hypersphere = F.normalize(self.con_proj_head(feat_mat_cls), dim=1)
                hypersphere_cc1 = F.normalize(self.con_proj_head(feat_mat_cls_cc1), dim=1)
                hypersphere_cc2 = F.normalize(self.con_proj_head(feat_mat_cls_cc2), dim=1)
                if return_ccrop_img:
                    return cls_pred, \
                           cls_pred_cc1, cls_pred_cc2, en_pred, \
                           hypersphere, hypersphere_cc1, hypersphere_cc2, \
                           cc_img_1, cc_img_2, cc_box1, cc_box2, \
                           atten_maps, atten_maps_cc1, atten_maps_cc2
                else:
                    return cls_pred, cls_pred_cc1, cls_pred_cc2, en_pred, hypersphere, hypersphere_cc1, hypersphere_cc2
            else:
                return cls_pred, en_pred
        elif phase == "inference":
            crop_images, height_min_ids, height_max_ids, width_min_ids, width_max_ids, = self.gen_Zoom_in(
                images=x.detach().clone(), atten_maps=atten_maps)
            attention_maps_cls2, feature_matrix_cls2 = self.eff_back(crop_images, return_cls_feat=True)

            if blocking:
                feat_mat_cls_ = feat_mat_cls.detach().clone()
                feat_mat_cls2_ = feature_matrix_cls2.detach().clone()
            else:
                feat_mat_cls_ = feat_mat_cls
                feat_mat_cls2_ = feature_matrix_cls2

            cls_pred = self.cls_head(feat_mat_cls_)
            cls_pred2 = self.cls_head(feat_mat_cls2_)
            en_pred = self.en_fc(torch.cat([feat_mat_cls_, feat_mat_cls2_], dim=1))

            if con_proj:
                hypersphere = F.normalize(self.con_proj_head(feat_mat_cls), dim=1)
                hypersphere2 = F.normalize(self.con_proj_head(feature_matrix_cls2), dim=1)
                if return_zoomin_img:

                    attention_maps_cls2 = torch.mean(attention_maps_cls2.detach(), dim=1, keepdim=True)
                    zoomin_box = torch.tensor([height_min_ids, height_max_ids, width_min_ids, width_max_ids]).T

                    return cls_pred, cls_pred2, en_pred, \
                           hypersphere, hypersphere2, \
                           crop_images, zoomin_box, \
                           att_maps_cls, attention_maps_cls2
                else:
                    return cls_pred, att_maps_cls, cls_pred2, en_pred, hypersphere, hypersphere2
            else:
                return cls_pred, att_maps_cls, cls_pred2, en_pred
        else:
            raise NotImplementedError(f"Not Implemented phase:{phase}")


if __name__ == "__main__":
    pass
