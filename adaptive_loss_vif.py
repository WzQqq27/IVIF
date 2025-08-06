from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
# from utils.utils_color import RGB_HSV, RGB_YCbCr
from loss_ssim import ssim
import torchvision.transforms.functional as TF

class calculate_ag(nn.Module):
    def __init__(self):
        super(calculate_ag, self).__init__()

    def forward(self, x):
        """计算平均梯度（Average Gradient, AG），用于调整SSIM权重"""
        gradient = self.sobel(x)
        ag = torch.mean(gradient, dim=[1, 2, 3], keepdim=True)
        return ag


class calculate_en(nn.Module):
    def __init__(self):
        super(calculate_en, self).__init__()

    def forward(self, x):
        """计算图像熵（Entropy, EN），用于调整MSE权重"""
        hist = torch.histc(x, bins=256, min=0.0, max=1.0) / x.numel()
        en = -torch.sum(hist * torch.log2(hist + 1e-7))
        return en


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        ag_A = self.calculate_ag(image_A)
        ag_B = self.calculate_ag(image_B)
        sigma_A = torch.exp(ag_A) / (torch.exp(ag_A) + torch.exp(ag_B))  # 公式(7)
        sigma_B = torch.exp(ag_B) / (torch.exp(ag_A) + torch.exp(ag_B))
        Loss_SSIM = (sigma_A * (1 - ssim(image_A, image_fused))
                     + sigma_B * (1 - ssim(image_B, image_fused)))  # 自适应SSIMreturn Loss_SSIM
        return Loss_SSIM
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        # MSE权重（基于图像熵EN）
        en_A = self.calculate_en(image_A)
        en_B = self.calculate_en(image_B)
        gamma_A = torch.exp(en_A) / (torch.exp(en_A) + torch.exp(en_B))  # 公式(9)
        gamma_B = torch.exp(en_B) / (torch.exp(en_A) + torch.exp(en_B))
        Loss_intensity = (gamma_A * F.l1_loss(image_fused, image_A)
                          + gamma_B * F.l1_loss(image_fused, image_B))
        return Loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)

    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 15 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

