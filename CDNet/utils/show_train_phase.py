import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def convert_tensor_to_visual_np_img(img:torch.Tensor):
    """
    用于将tensor数据转换为np格式的数据，可直接放在matplotlib中可视化展示，
    此处的tensor经过 transform中 mean=0.5,std=0.5的调整，无法直接可视化
    :param img: C * H * W, 一般为 3 * 224 * 224
    :return: img_np，经过还原的图像，该图像格式为 uint8
    """
    assert len(img.shape) == 3, f"Invalid img with shape:{img.shape}"

    img_np = img.to("cpu").permute(1, 2, 0).numpy()
    img_np = img_np * 127.5 + 127.5
    img_np = (img_np / img_np.max()) * 255
    img_np = img_np.astype("uint8")
    img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

    return img_np


def visual_attention_map(img:torch.Tensor,attention_map:torch.Tensor,img_size:int=224,B:int=40,G:int=87,R:int=100):
    """

    :param img:
    :param attention_map:
    :param img_size:
    :param B:
    :param G:
    :param R:
    :return:
    """
    assert len(img.shape) == len(attention_map.shape) == 3, \
        f"Invalid img shape{img.shape} and att shape:{attention_map.shape}"

    img = convert_tensor_to_visual_np_img(img=img)
    vis_mask_B = np.full(shape=(img_size, img_size, 1), fill_value=B)
    vis_mask_G = np.full(shape=(img_size, img_size, 1), fill_value=G)
    vis_mask_R = np.full(shape=(img_size, img_size, 1), fill_value=R)
    vis_mask = np.concatenate([vis_mask_B, vis_mask_G, vis_mask_R], axis=2)

    theta_c = 0.6 * attention_map.max()

    crop_mask = F.interpolate(attention_map.unsqueeze(0), size=(224, 224), mode='bilinear') >= theta_c
    nonzero_indices = torch.nonzero(crop_mask[0, 0, :, :], as_tuple=False)
    height_min = max(int(nonzero_indices[:, 0].min().item()), 0)
    height_max = min(int(nonzero_indices[:, 0].max().item()), img_size)
    width_min = max(int(nonzero_indices[:, 1].min().item()), 0)
    width_max = min(int(nonzero_indices[:, 1].max().item()), img_size)


    crop_mask = crop_mask.to("cpu").squeeze(0).permute(1, 2, 0).numpy()
    img_vis = crop_mask * vis_mask + img
    img_vis = (img_vis / img_vis.max()) * 255.
    img_vis = img_vis.astype(np.uint8)
    img_vis = cv.cvtColor(img_vis, cv.COLOR_BGR2RGB)

    return img_vis, height_min, height_max, width_min, width_max


def save_images_MajorRevision(ori_img, GT_text, ori_att, PR_text, zom_img, zom_att, save_path, img_name, save_util="plt"):
    """

    :param ori_img:
    :param GT_text:
    :param ori_att:
    :param PR_text:
    :param zom_img:
    :param zom_att:
    :param save_path:
    :param img_name:
    :param save_util:
    :return:
    """
    if save_util == "plt":
        plt.figure(figsize=(8,8),dpi=100)
        plt.subplot(2, 2, 1)
        plt.title(f"OriImage,GT:{GT_text}")
        plt.imshow(ori_img)

        plt.subplot(2, 2, 2)
        plt.title(f"OriImgAtt,PR:{PR_text}")
        plt.imshow(ori_att)

        plt.subplot(2, 2, 3)
        plt.title("CropImg")
        plt.imshow(zom_img)

        plt.subplot(2, 2, 4)
        plt.title("CropImg Att")
        plt.imshow(zom_att)

        plt.savefig(os.path.join(save_path, f"{img_name}_img_all_in.png"), bbox_inches='tight')
        plt.close()
    elif save_util == "cv":
        save_path = os.path.join(save_path, f"{img_name}_img_all_in")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv.imwrite(os.path.join(save_path,"ori_img.png"), cv.cvtColor(ori_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"ori_att.png"), cv.cvtColor(ori_att,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"zom_img.png"), cv.cvtColor(zom_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"zom_att.png"), cv.cvtColor(zom_att,cv.COLOR_RGB2BGR))

    else:
        raise NotImplementedError(f"Not implemented {save_util}")


def save_att_images(ori_att, save_path, img_name):
    """

    :param ori_att:
    :param save_path:
    :param img_name:
    :return:
    """
    save_path = os.path.join(save_path, f"{img_name}_att_all_in")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ori_att = F.interpolate(ori_att, size=(224, 224), mode='bilinear').to("cpu").squeeze().numpy()
    ori_att = ori_att / np.max(ori_att)
    for i, att in enumerate(ori_att):
        plt.imshow(att)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.savefig(os.path.join(save_path, f"att_{i}.png"), bbox_inches='tight')
        plt.close()


def save_center_cmz_images(ori_img, ori_att, cen_img, cen_att, cmz1_img, cmz1_att, cmz2_img, cmz2_att, save_path, img_name, save_util="plt"):
    """

    :param ori_img:
    :param ori_att:
    :param cen_img:
    :param cen_att:
    :param cmz1_img:
    :param cmz1_att:
    :param cmz2_img:
    :param cmz2_att:
    :param save_path:
    :param img_name:
    :param save_util:
    :return:
    """
    if save_util == "plt":
        plt.figure(figsize=(16,8),dpi=100)
        plt.subplot(2, 4, 1)
        # plt.title("Original")
        plt.imshow(ori_img)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 2)
        # plt.title("Center")
        plt.imshow(cen_img)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 3)
        # plt.title("CMZ1")
        plt.imshow(cmz1_img)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 4)
        # plt.title("CMZ2")
        plt.imshow(cmz2_img)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 5)
        # plt.title("Att image")
        plt.imshow(ori_att)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 6)
        # plt.title("Att center")
        plt.imshow(cen_att)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 7)
        # plt.title("Att cmz-1")
        plt.imshow(cmz1_att)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        plt.subplot(2, 4, 8)
        # plt.title("Att cmz-2")
        plt.imshow(cmz2_att)

        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.savefig(os.path.join(save_path, f"{img_name}_cen_cmz_all_in.png"), bbox_inches='tight')
        plt.close()
    elif save_util == "cv":
        save_path = os.path.join(save_path, f"{img_name}_img_all_in")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv.imwrite(os.path.join(save_path,"1_ori_img.png"), cv.cvtColor(ori_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"2_cen_img.png"), cv.cvtColor(cen_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"3_cmz_img.png"), cv.cvtColor(cmz1_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"4_cmz_img.png"), cv.cvtColor(cmz2_img,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"5_ori_att.png"), cv.cvtColor(ori_att,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"6_cen_att.png"), cv.cvtColor(cen_att,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"7_cmz_att.png"), cv.cvtColor(cmz1_att,cv.COLOR_RGB2BGR))
        cv.imwrite(os.path.join(save_path,"8_cmz_att.png"), cv.cvtColor(cmz2_att,cv.COLOR_RGB2BGR))
    else:
        raise NotImplementedError(f"Not implemented {save_util}")