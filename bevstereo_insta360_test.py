import sys

device = sys.argv[1]
scene_dir = sys.argv[2]

if len(sys.argv) != 3:
    print("Insufficient arguments")
    sys.exit()

print("device: " + device)
print("scene: " + scene_dir)

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= device

from functools import partial
import numba
import math
import numpy as np
from scipy.special import erf
from scipy.stats import norm

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models as models
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset

import torch.utils.model_zoo as model_zoo

import mmcv
from mmcv.cnn import build_conv_layer
from mmcv.runner import load_checkpoint

from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.core import reduce_mean

from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models import build_neck
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead, circle_nms
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models import build_neck

from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from nuscenes.utils.data_classes import Box
from bevdepth.callbacks.ema import EMACallback
from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize

try:
    from bevdepth.ops.voxel_pooling import voxel_pooling
except ImportError:
    print('Import VoxelPooling fail.')

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image
from pyquaternion import Quaternion
import pyquaternion

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import Box, LidarPointCloud
from collections import OrderedDict

# from bevdepth.datasets.insta360_det_dataset import map_name_from_general_to_detection
# from bevdepth.datasets.insta360_det_dataset import NuscDetDataset, collate_fn

import glob
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error




# device = torch.device('cuda:1')
# torch.cuda.set_device(device)




infos_path = '../data/nuscenes/nuscenes_infos_val.pkl'
infos = mmcv.load(infos_path)

"""from layers/backbones/base_lss_fpn"""

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x

class BaseLSSFPN(nn.Module):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 img_backbone_conf,
                 img_neck_conf,
                 depth_net_conf,
                 use_da=True): # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 임의로 바꿈
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(BaseLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = self._configure_depth_aggregation_net()
            
    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels, self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum: downsample 하는 것을 고려해서 frustum 만듦 (final dim에 맞게)"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        """ frustum은 704, 256에 맞게 생성되었는데, 
        frustum 상의 camera coord. points를 ego coord.로 정확히 옮기기 위해
        ida 적용 이전 원래대로 돌려놓고 camera coord. => ego coord. 수행 """
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)
        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        depth = depth_feature[:, :self.depth_channels].softmax(1)
        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels)].unsqueeze(2)

        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        
        feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.cuda())
        
        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_depth=is_return_depth)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)

        
"""from layers/backbones/bevstereo_lss_fpn.py"""

class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D +
            batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 d_bound,
                 num_ranges=4):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_feat_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.mu_sigma_range_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      num_ranges * 3,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.mono_depth_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.d_bound = d_bound
        self.num_ranges = num_ranges

    # @autocast(False)
    def forward(self, x, mats_dict, scale_depth_factor=1000.0):
        B, _, H, W = x.shape
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(x, depth_se)
        depth_feat = self.depth_feat_conv(depth_feat)
        mono_depth = self.mono_depth_net(depth_feat)
        mu_sigma_score = self.mu_sigma_range_net(depth_feat)
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).reshape(1, -1, 1, 1).cuda()
        d_coords = d_coords.repeat(B, 1, H, W)
        mu = mu_sigma_score[:, 0:self.num_ranges, ...]
        sigma = mu_sigma_score[:, self.num_ranges:2 * self.num_ranges, ...]
        range_score = mu_sigma_score[:,
                                     2 * self.num_ranges:3 * self.num_ranges,
                                     ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return x, context, mu, sigma, range_score, mono_depth


class BEVStereoLSSFPN(BaseLSSFPN):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 img_backbone_conf,
                 img_neck_conf,
                 depth_net_conf,
                 use_da=False,
                 sampling_range=3,
                 num_samples=3,
                 stereo_downsample_factor=4,
                 em_iteration=3,
                 min_sigma=1,
                 num_groups=8,
                 num_ranges=4,
                 range_list=[[2, 8], [8, 16], [16, 28], [28, 58]],
                 k_list=None,
                 use_mask=True):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.
        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
            sampling_range (int): The base range of sampling candidates.
                Defaults to 3.
            num_samples (int): Number of samples. Defaults to 3.
            stereo_downsample_factor (int): Downsample factor from input image
                and stereo depth. Defaults to 4.
            em_iteration (int): Number of iterations for em. Defaults to 3.
            min_sigma (float): Minimal value for sigma. Defaults to 1.
            num_groups (int): Number of groups to keep after inner product.
                Defaults to 8.
            num_ranges (int): Number of split ranges. Defaults to 1.
            range_list (list): Start and end of every range, Defaults to None.
            k_list (list): Depth of all candidates inside the range.
                Defaults to None.
            use_mask (bool): Whether to use mask_net. Defaults to True.
        """
        self.num_ranges = num_ranges
        self.sampling_range = sampling_range
        self.num_samples = num_samples
        super(BEVStereoLSSFPN,
              self).__init__(x_bound, y_bound, z_bound, d_bound, final_dim,
                             downsample_factor, output_channels,
                             img_backbone_conf, img_neck_conf, depth_net_conf,
                             use_da)

        self.depth_channels, _, _, _ = self.frustum.shape
        self.use_mask = use_mask
        if k_list is None:
            self.register_buffer('k_list', torch.Tensor(self.depth_sampling()))
        else:
            self.register_buffer('k_list', torch.Tensor(k_list))
        self.stereo_downsample_factor = stereo_downsample_factor
        self.em_iteration = em_iteration
        self.register_buffer(
            'depth_values',
            torch.arange((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2],
                         dtype=torch.float))
        self.num_groups = num_groups
        self.similarity_net = nn.Sequential(
            ConvBnReLU3D(in_channels=num_groups,
                         out_channels=16,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            ConvBnReLU3D(in_channels=16,
                         out_channels=8,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        if range_list is None:
            range_length = (d_bound[1] - d_bound[0]) / num_ranges
            self.range_list = [[
                d_bound[0] + range_length * i,
                d_bound[0] + range_length * (i + 1)
            ] for i in range(num_ranges)]
        else:
            assert len(range_list) == num_ranges
            self.range_list = range_list

        self.min_sigma = min_sigma
        self.depth_downsample_net = nn.Sequential(
            nn.Conv2d(self.depth_channels, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.depth_channels, 1, 1, 0),
        )
        self.context_downsample_net = nn.Identity()
        if self.use_mask:
            self.mask_net = nn.Sequential(
                nn.Conv2d(224, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.Sigmoid(),
            )

    def depth_sampling(self):
        """Generate sampling range of candidates.
        Returns:
            list[float]: List of all candidates.
        """
        P_total = erf(self.sampling_range / np.sqrt(2))  # Probability covered by the sampling range
        idx_list = np.arange(0, self.num_samples + 1)
        p_list = (1 - P_total) / 2 + ((idx_list / self.num_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return list(k_list)

    def _generate_cost_volume(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Generate cost volume based on depth sample.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Depth score for all sweeps.
        """
        batch_size, num_channels, height, width = stereo_feats_all_sweeps[0].shape
        num_sweeps = len(stereo_feats_all_sweeps)
        depth_score_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index: # 같은 시점의 frames (똑같은 두 frames)로는 cost volume 구축하지 않도록 (src <=> ref 사이에서만 구축)
                continue
            
            warped_stereo_fea = self.homo_warping(
                stereo_feats_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(stereo_feats_all_sweeps[idx]),
            )
            warped_stereo_fea = warped_stereo_fea.reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                self.num_samples, height, width)
            ref_stereo_feat = stereo_feats_all_sweeps[sweep_index].reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                height, width)
            feat_cost = torch.mean(
                (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)
        return torch.stack(depth_score_all_sweeps).mean(0)

    def homo_warping(
        self,
        stereo_feat,
        key_intrin_mats,
        sweep_intrin_mats,
        sensor2sensor_mats,
        key_ida_mats,
        sweep_ida_mats,
        depth_sample,
        frustum,
    ):
        """Used for mvs method to transfer sweep image feature to
            key image feature.
        Args:
            src_fea(Tensor): image features.
            key_intrin_mats(Tensor): Intrin matrix for key sensor.
            sweep_intrin_mats(Tensor): Intrin matrix for sweep sensor.
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            key_ida_mats(Tensor): Ida matrix for key frame.
            sweep_ida_mats(Tensor): Ida matrix for sweep frame.
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
        """
        batch_size_with_num_cams, channels = stereo_feat.shape[
            0], stereo_feat.shape[1]
        height, width = stereo_feat.shape[2], stereo_feat.shape[3]
        with torch.no_grad():
            points = frustum
            points = points.reshape(points.shape[0], -1, points.shape[-1])
            points[..., 2] = 1
            # Undo ida for key frame.
            points = key_ida_mats.reshape(batch_size_with_num_cams, *
                                          key_ida_mats.shape[2:]).inverse(
                                          ).unsqueeze(1) @ points.unsqueeze(-1)
            # Convert points from pixel coord to key camera coord.
            points[..., :3, :] *= depth_sample.reshape(batch_size_with_num_cams, -1, 1, 1)
            num_depth = frustum.shape[1]
            points = (key_intrin_mats.reshape(
                batch_size_with_num_cams, *
                key_intrin_mats.shape[2:]).inverse().unsqueeze(1) @ points)
            points = (sensor2sensor_mats.reshape(
                batch_size_with_num_cams, *
                sensor2sensor_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep sensor coord.
            points = (sweep_intrin_mats.reshape(
                batch_size_with_num_cams, *
                sweep_intrin_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep pixel coord.
            points[..., :2, :] = points[..., :2, :] / points[..., 2:3, :]  # [B, 2, Ndepth, H*W]

            points = (sweep_ida_mats.reshape(
                batch_size_with_num_cams, *
                sweep_ida_mats.shape[2:]).unsqueeze(1) @ points).squeeze(-1)
            neg_mask = points[..., 2] < 1e-3
            points[..., 0][neg_mask] = width * self.stereo_downsample_factor
            points[..., 1][neg_mask] = height * self.stereo_downsample_factor
            points[..., 2][neg_mask] = 1
            proj_x_normalized = points[..., 0] / ((width * self.stereo_downsample_factor - 1) / 2) - 1
            proj_y_normalized = points[..., 1] / ((height * self.stereo_downsample_factor - 1) / 2) - 1
            grid = torch.stack([proj_x_normalized, proj_y_normalized], dim=2)  # [B, Ndepth, H*W, 2]

        warped_stereo_fea = F.grid_sample(
            stereo_feat,
            grid.view(batch_size_with_num_cams, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
        )
        warped_stereo_fea = warped_stereo_fea.view(batch_size_with_num_cams,
                                                   channels, num_depth, height,
                                                   width)

        return warped_stereo_fea

    def _forward_stereo(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mono_depth_all_sweeps,
        mats_dict,
        sensor2sensor_mats,
        mu_all_sweeps,
        sigma_all_sweeps,
        range_score_all_sweeps,
        depth_feat_all_sweeps,
    ):
        """Forward function to generate stereo depth.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature of all sweeps.
            mono_depth_all_sweeps (list[Tensor]):
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            mu_all_sweeps (list[Tensor]): List of mu for all sweeps.
            sigma_all_sweeps (list[Tensor]): List of sigma for all sweeps.
            range_score_all_sweeps (list[Tensor]): List of all range score
                for all sweeps.
            depth_feat_all_sweeps (list[Tensor]): List of all depth feat for
                all sweeps.
        Returns:
            Tensor: stereo_depth
        """
        batch_size_with_cams, _, feat_height, feat_width = stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float,
                                device=device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(batch_size_with_cams, 1, feat_height,
                                   feat_width)
        stereo_depth = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams, self.depth_channels, feat_height, feat_width)
        mask_score = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams,
            self.depth_channels,
            feat_height * self.stereo_downsample_factor //
            self.downsample_factor,
            feat_width * self.stereo_downsample_factor //
            self.downsample_factor,
        )
        score_all_ranges = list()
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        
        for range_idx in range(self.num_ranges):
            # Map mu to the corresponding interval.
            range_start = self.range_list[range_idx][0]
            mu_all_sweeps_single_range = [
                mu[:, range_idx:range_idx + 1, ...].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_all_sweeps_single_range = [
                sigma[:, range_idx:range_idx + 1, ...]
                for sigma in sigma_all_sweeps
            ]
            batch_size_with_cams, _, feat_height, feat_width = stereo_feats_all_sweeps[0].shape
            mu = mu_all_sweeps_single_range[sweep_index]
            sigma = sigma_all_sweeps_single_range[sweep_index]
            for _ in range(self.em_iteration):
                depth_sample = torch.cat([mu + sigma * k for k in self.k_list], 1)
                depth_sample_frustum = self.create_depth_sample_frustum(depth_sample, self.stereo_downsample_factor)
                
                """
                cost volumn 함수에 stereo features와 depth sample (mu와 sigma에 대한 함수의 아웃풋)가 input으로 들어감
                mu와 sigma는 그럼 source image features에서 예측된 값? YES!
                
                BEVStereo의 key idea는 cost volumn을 구축할 때 모든 depth samples (depth map의 모든 pixels)를 고려하는 것이 아닌,
                일부 depth samples (gen from mu and sigma)로부터 sparse cost volumn을 구축함!
                """
                
                mu_score = self._generate_cost_volume(
                    sweep_index,
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples // 2:self.num_samples // 2 + 1, ...]),
                    min=0.1,
                    max=10)

                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum
            range_length = int((self.range_list[range_idx][1] - self.range_list[range_idx][0]) // self.d_bound[2])
            if self.use_mask:
                depth_sample = F.avg_pool2d(
                    mu,
                    self.downsample_factor // self.stereo_downsample_factor,
                    self.downsample_factor // self.stereo_downsample_factor,
                )
                depth_sample_frustum = self.create_depth_sample_frustum(depth_sample, self.downsample_factor)
                mask = self._forward_mask(
                    sweep_index,
                    mono_depth_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mask_score[:,
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]):range_length +
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]), ..., ] += mask
                del depth_sample
                del depth_sample_frustum
            sigma = torch.clamp(sigma, self.min_sigma)
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.d_bound[0]) //
                              self.d_bound[2]):range_length + int(
                                  (range_start - self.d_bound[0]) //
                                  self.d_bound[2]), ..., ] - mu_repeated) /
                torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            score_all_ranges.append(mu_score.sum(1).unsqueeze(1))
            depth_score_single_range = depth_score_single_range / (sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:,
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]):range_length +
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]), ..., ] = (
                                 depth_score_single_range *
                                 range_score[:, range_idx:range_idx + 1, ...])
            del depth_score_single_range
            del mu_repeated
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth

    def create_depth_sample_frustum(self, depth_sample, downsample_factor=16):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        x_coords = (torch.linspace(0,
                                   ogfW - 1,
                                   fW,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(1, 1, 1, fW).expand(batch_size, num_depth, fH, fW))
        y_coords = (torch.linspace(0,
                                   ogfH - 1,
                                   fH,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(1, 1, fH, 1).expand(batch_size, num_depth, fH, fW))
        paddings = torch.ones_like(depth_sample)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, depth_sample, paddings), -1)
        return frustum

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            self.d_bound,
            self.num_ranges,
        )

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(backbone_feats)[0]
        img_feats_reshape = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                              img_feats.shape[1],
                                              img_feats.shape[2],
                                              img_feats.shape[3])
        return img_feats_reshape, backbone_feats[0].detach()

    def _forward_mask(
        self,
        sweep_index,
        mono_depth_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Forward function to generate mask.
        Args:
            sweep_index (int): Index of sweep.
            mono_depth_all_sweeps (list[Tensor]): List of mono_depth for
                all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Generated mask.
        """
        num_sweeps = len(mono_depth_all_sweeps)
        mask_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_mono_depth = self.homo_warping(
                mono_depth_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(mono_depth_all_sweeps[idx]),
            )
            mask = self.mask_net(
                torch.cat([
                    mono_depth_all_sweeps[sweep_index].detach(),
                    warped_mono_depth.mean(2).detach()
                ], 1))
            mask_all_sweeps.append(mask)
        return torch.stack(mask_all_sweeps).mean(0)

    def _forward_single_sweep(self,
                              sweep_index,
                              context,
                              mats_dict,
                              depth_score,
                              is_return_depth=False):
        """Forward function for single sweep.
        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.
        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_cams = context.shape[0], context.shape[1]
        context = context.reshape(batch_size * num_cams, *context.shape[2:])
        depth = depth_score
        img_feat_with_depth = depth.unsqueeze(1) * context.unsqueeze(2)

        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        
        feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous().float(), self.voxel_num.cuda())
        
        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                
                # for pose network
                posenet_outputs=None,
                
                is_return_depth=False):
        """Forward function.
        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).
        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        context_all_sweeps = list()
        depth_feat_all_sweeps = list()
        img_feats_all_sweeps = list()
        stereo_feats_all_sweeps = list()
        mu_all_sweeps = list()
        sigma_all_sweeps = list()
        mono_depth_all_sweeps = list()
        range_score_all_sweeps = list()
        
        """ num_sweep: 2 (연속된 두 개의 keyframes) """ 
        
        """ 1. Mono depth map (mu, sigma 포함)과 image features 추출 """
        for sweep_index in range(0, num_sweeps):
            if sweep_index > 0:
                with torch.no_grad():
                    img_feats, stereo_feats = self.get_cam_feats(sweep_imgs[:, sweep_index:sweep_index + 1, ...])
                    
                    img_feats_all_sweeps.append(img_feats.view(batch_size * num_cams, *img_feats.shape[3:]))
                    stereo_feats_all_sweeps.append(stereo_feats)
                    
                    depth_feat, context, mu, sigma, range_score, mono_depth =\
                        self.depth_net(img_feats.view(batch_size * num_cams, *img_feats.shape[3:]), mats_dict)
                    
                    context_all_sweeps.append(
                        self.context_downsample_net(
                            context.reshape(batch_size * num_cams, *context.shape[1:])))
                    depth_feat_all_sweeps.append(depth_feat)
            else:
                img_feats, stereo_feats = self.get_cam_feats(sweep_imgs[:, sweep_index:sweep_index + 1, ...])
                
                img_feats_all_sweeps.append(img_feats.view(batch_size * num_cams, *img_feats.shape[3:]))
                stereo_feats_all_sweeps.append(stereo_feats)
                
                depth_feat, context, mu, sigma, range_score, mono_depth =\
                    self.depth_net(img_feats.view(batch_size * num_cams, *img_feats.shape[3:]), mats_dict)
                
                context_all_sweeps.append(
                    self.context_downsample_net(
                        context.reshape(batch_size * num_cams, *context.shape[1:])))
                depth_feat_all_sweeps.append(depth_feat)
                
            mu_all_sweeps.append(mu)
            sigma_all_sweeps.append(sigma)
            mono_depth_all_sweeps.append(mono_depth)
            range_score_all_sweeps.append(range_score)
        
        """ 2. Stereo depth 추출 (_forward_stereo() function) """
        depth_score_all_sweeps = list()
        for ref_idx in range(num_sweeps):
            sensor2sensor_mats = list()
            for src_idx in range(num_sweeps):
                
                """ Original Code """
                # ref2keysensor_mats = mats_dict['sensor2sensor_mats'][:, ref_idx, ...].inverse()
                # key2srcsensor_mats = mats_dict['sensor2sensor_mats'][:, src_idx, ...]
                # ref2srcsensor_mats = key2srcsensor_mats @ ref2keysensor_mats
                # sensor2sensor_mats.append(ref2srcsensor_mats)
                
                """ Modification for PoseNet """
                if posenet_outputs is not None:
                    print("Use posenet output")
                    if ref_idx < src_idx:
                        ref2srcsensor_mats = posenet_outputs[0]
                    else:
                        ref2srcsensor_mats = posenet_outputs[1]
                else:
                    print("Use Nusc pose data")
                    ref2keysensor_mats = mats_dict['sensor2sensor_mats'][:, ref_idx, ...].inverse()
                    key2srcsensor_mats = mats_dict['sensor2sensor_mats'][:, src_idx, ...]
                    ref2srcsensor_mats = key2srcsensor_mats @ ref2keysensor_mats
                sensor2sensor_mats.append(ref2srcsensor_mats)
                # if ref_idx != src_idx:
                #     print("ref_idx: ", ref_idx, " src_idx: ", src_idx)
                #     print("ref2srcsensor_mats: \n", ref2srcsensor_mats)
                #     print("shape: ", ref2srcsensor_mats.shape)
                
            if ref_idx == 0:
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                if self.use_mask:
                    stereo_depth, mask = self._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
                else:
                    stereo_depth = self._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
            else:
                with torch.no_grad():
                    # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                    if self.use_mask:
                        stereo_depth, mask = self._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
                    else:
                        stereo_depth = self._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
            if self.use_mask:
                depth_score = (
                    mono_depth_all_sweeps[ref_idx] +
                    self.depth_downsample_net(stereo_depth) * mask).softmax(1)
            else:
                depth_score = (
                    mono_depth_all_sweeps[ref_idx] +
                    self.depth_downsample_net(stereo_depth)).softmax(1)
                
            depth_score_all_sweeps.append(depth_score)
        
        """ 3. Stereo depth map과 context image feature로 voxel pooling하여 BEV features를 얻음 """ 
        """    3-1. 현재 시점의 keyframe에 대해서 voxel pooling """ 
        key_frame_res = self._forward_single_sweep(
            0,
            context_all_sweeps[0].reshape(batch_size, num_cams, *context_all_sweeps[0].shape[1:]),
            mats_dict,
            depth_score_all_sweeps[0],
            is_return_depth=is_return_depth,
        )
        if num_sweeps == 1:
            return key_frame_res
        
        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res
        ret_feature_list = [key_frame_feature]
        
        """    3-2. 다음 시점의 keyframe에 대해서 voxel pooling """
        for sweep_index in range(1, num_sweeps): # num_sweep: 2 -> one iteration (i==1)
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index, # 1
                    context_all_sweeps[sweep_index].reshape(batch_size, num_cams, *context_all_sweeps[sweep_index].shape[1:]),
                    mats_dict,
                    depth_score_all_sweeps[sweep_index],
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), depth_score_all_sweeps[0]
        else:
            return torch.cat(ret_feature_list, 1)

bev_backbone_conf = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(type='SECONDFPN',
                     in_channels=[160, 320, 640],
                     upsample_strides=[2, 4, 8],
                     out_channels=[64, 64, 128])

@numba.jit(nopython=True)
def size_aware_circle_nms(dets, thresh_scale, post_max_size=83):
    """Circular NMS.
    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.
    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83
    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    dx1 = dets[:, 2]
    dy1 = dets[:, 3]
    yaws = dets[:, 4]
    scores = dets[:, -1]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist_x = abs(x1[i] - x1[j])
            dist_y = abs(y1[i] - y1[j])
            dist_x_th = (abs(dx1[i] * np.cos(yaws[i])) +
                         abs(dx1[j] * np.cos(yaws[j])) +
                         abs(dy1[i] * np.sin(yaws[i])) +
                         abs(dy1[j] * np.sin(yaws[j])))
            dist_y_th = (abs(dx1[i] * np.sin(yaws[i])) +
                         abs(dx1[j] * np.sin(yaws[j])) +
                         abs(dy1[i] * np.cos(yaws[i])) +
                         abs(dy1[j] * np.cos(yaws[j])))
            # ovr = inter / areas[j]
            if dist_x <= dist_x_th * thresh_scale / 2 and \
               dist_y <= dist_y_th * thresh_scale / 2:
                suppressed[j] = 1
    return keep[:post_max_size]

class BEVDepthHead(CenterHead):
    """Head for BevDepth.

    Args:
        in_channels(int): Number of channels after bev_neck.
        tasks(dict): Tasks for head.
        bbox_coder(dict): Config of bbox coder.
        common_heads(dict): Config of head for each task.
        loss_cls(dict): Config of classification loss.
        loss_bbox(dict): Config of regression loss.
        gaussian_overlap(float): Gaussian overlap used for `get_targets`.
        min_radius(int): Min radius used for `get_targets`.
        train_cfg(dict): Config used in the training process.
        test_cfg(dict): Config used in the test process.
        bev_backbone_conf(dict): Cnfig of bev_backbone.
        bev_neck_conf(dict): Cnfig of bev_neck.
    """

    def __init__(
        self,
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=bev_backbone_conf,
        bev_neck_conf=bev_neck_conf,
        separate_head=dict(type='SeparateHead',
                           init_bias=-2.19,
                           final_kernel=3),
    ):
        super(BEVDepthHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
        )
        self.trunk = build_backbone(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = build_neck(bev_neck_conf)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @autocast(False)
    def forward(self, x):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        # FPN
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = super().forward(fpn_output)
        return ret_values

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            anno_box = gt_bboxes_3d.new_zeros(
                (max_objs, len(self.train_cfg['code_weights'])),
                dtype=torch.float32,
                device='cuda')

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device='cuda')
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = targets
        return_loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict[0].keys():
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel']),
                    dim=1,
                )
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot']),
                    dim=1,
                )
            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)
            return_loss += loss_bbox
            return_loss += loss_heatmap
        return return_loss

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(batch_heatmap,
                                          batch_rots,
                                          batch_rotc,
                                          batch_hei,
                                          batch_dim,
                                          batch_vel,
                                          reg=batch_reg,
                                          task_id=task_id)
            assert self.test_cfg['nms_type'] in [
                'size_aware_circle', 'circle', 'rotate'
            ]
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size']),
                                        dtype=torch.long,
                                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            elif self.test_cfg['nms_type'] == 'size_aware_circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    boxes_2d = boxes3d[:, [0, 1, 3, 4, 6]]
                    boxes = torch.cat([boxes_2d, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        size_aware_circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['thresh_scale'][task_id],
                            post_max_size=self.test_cfg['post_max_size'],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

class BEVStereo(nn.Module):
    """Source code of `BEVStereo`, `https://arxiv.org/abs/2209.10248`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BEVStereo, self).__init__()
        self.backbone = BEVStereoLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth
    
    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
        
        # for pose network
        posenet_outputs=None
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps, 
                              # for pose network
                              posenet_outputs=posenet_outputs)
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels': 80,
    'downsample_factor': 16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim': final_dim,
    'rot_lim': (-5.4, 5.4),
    'H': H,
    'W': W,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }



def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)
    
    # print("R: ", R)
    # print("R[:3, :3]: ", R[:, :3, :3])
    # print("T: ", T)
    # print("T[~]: ", T[:, :3, -1])
    
    # R, T를 이어 붙여서 return
    M = R.new_zeros((4, 4))
    M[3, 3] = 1
    M[:3, :3] = R[:, :3, :3]
    M[:3, -1] = T[:, :3, -1]

    # R, T 간의 행렬곱을 return
    # if invert:
    #     M = torch.matmul(R, T)
    # else:
    #     M = torch.matmul(T, R)

    return M

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

"""Pose Encoder"""
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder"""
    def __init__(self, num_layers, pretrained, num_input_images=1, **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

"""Pose Decoder"""
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

def get_quaternion_from_euler(e):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    roll = e[0]
    pitch = e[1]
    yaw = e[2]

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def euler_from_quaternion(q):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    import math
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1) / np.pi * 180 # degrees
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2) / np.pi * 180
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z) 
    # yaw_z = math.atan2(t3, t4) / np.pi * 180
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z] # in radian

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

""" test일 때는 flip 및 rotate하는 augmentation은 하지 않고, 오직 resize와 crop만 적용됨 """ 
def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize # resize == 704 / 1600 == 0.44
    ida_tran -= torch.Tensor(crop[:2]) # crop: (0, 140, 704, 396), crop[:2]: (0, 140) => 즉 tran은 y축 방향으로 -140
    if flip: # test time에는 X
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi) # 0일때 (test) identity matrix
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2 # [704, 256] / 2 = [352, 128]
    b = A.matmul(-b) + b # A는 단위행렬이므로, 결국엔 -b + b, 즉 영행렬
    ida_rot = A.matmul(ida_rot) # [[0.44, 0], [0, 0.44]]
    ida_tran = A.matmul(ida_tran) + b # A @ ida_tran + b => tensor([   0., -140.])
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    # ida_mat
    # tensor([[   0.4400,    0.0000,    0.0000,    0.0000],
    #         [   0.0000,    0.4400,    0.0000, -140.0000],
    #         [   0.0000,    0.0000,    1.0000,    0.0000],
    #         [   0.0000,    0.0000,    0.0000,    1.0000]])
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring

def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    
    # for pose network
    pose_input_imgs_batch = list()
    
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
            
            # for pose network
            pose_input_imgs
        ) = iter_data[:11]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
            
        imgs_batch.append(sweep_imgs)
        
        # for pose network
        pose_input_imgs_batch.append(pose_input_imgs)
        
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
        
        # for pose network
        pose_input_imgs_batch
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    return ret_list

class NuscDetDataset(Dataset):

    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 
                 # Dataset customization
                 tangent_intrinsics=None,
                 sensor2ego_rot_eulers=None,
                 sensor2ego_trans=None,
                 ego2global_rotation=None,
                 ego2global_translation=None,
                 
                 infos=None,
                 
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 return_depth=False,
                 sweep_idxes=list(),
                 key_idxes=list(),
                 use_fusion=False):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
            use_fusion (bool): Whether to use lidar data.
                default: False.
        """
        super().__init__()
        if infos is None:
            if isinstance(info_paths, list):
                self.infos = list()
                for info_path in info_paths:
                    self.infos.extend(mmcv.load(info_path))
            else:
                self.infos = mmcv.load(info_paths)
        else:
            self.infos = infos
        
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes
        self.use_fusion = use_fusion
        
        # Dataset customization
        self.tangent_intrinsics = tangent_intrinsics
        self.sensor2ego_rot_eulers = sensor2ego_rot_eulers
        self.sensor2ego_trans = sensor2ego_trans
        self.ego2global_rotation = ego2global_rotation
        self.ego2global_translation = ego2global_translation

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)

    def get_image(self, cam_infos, cams, lidar_infos=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        
        # for pose network
        pose_input_imgs = list()
        
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            
            # for pose network
            campose_imgs = list()
            
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()
            for sweep_idx, cam_info in enumerate(cam_infos): # key_idxes [0, -1] 즉 현재 frmae과 직전 keyframe에 대해 enum

                img = Image.open(os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                
                # print("original", img.size)
                # display(img)
                
                # sweep sensor to sweep ego
                """변경. sensor2ego """
                if self.sensor2ego_rot_eulers is None:
                    w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation'] # 기존
                else:
                    sensor2ego_degrees = self.sensor2ego_rot_eulers[cam]
                    sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
                    sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
                    w, x, y, z = sensor2ego_q

                sweepsensor2sweepego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
                
                if self.sensor2ego_trans is None:
                    sweepsensor2sweepego_tran = torch.Tensor(cam_info[cam]['calibrated_sensor']['translation'])
                else:
                    sweepsensor2sweepego_tran = torch.Tensor(self.sensor2ego_trans) # 변경
                
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                """변경. sensor2ego """
                if self.sensor2ego_rot_eulers is None:
                    w, x, y, z = key_info[cam]['calibrated_sensor']['rotation'] # 기존
                else:
                    sensor2ego_degrees = self.sensor2ego_rot_eulers[cam]
                    sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
                    sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
                    w, x, y, z = sensor2ego_q
                keysensor2keyego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
                
                if self.sensor2ego_trans is None:
                    keysensor2keyego_tran = torch.Tensor(key_info[cam]['calibrated_sensor']['translation'])
                else:
                    keysensor2keyego_tran = torch.Tensor(self.sensor2ego_trans) # 변경
                
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                
                """ final value (matrix) """
                sweepsensor2keyego = global2keyego @ sweepego2global @ sweepsensor2sweepego # sensor2ego
                keysensor2sweepsensor = (keyego2keysensor @ sweepsensor2keyego).inverse() # sensor2sensor
                # keysensor2sweepsensor = (keyego2keysensor @ global2keyego @ sweepego2global @ sweepsensor2sweepego).inverse()
                
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                
                """ 변경. intrinsics """
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                if self.tangent_intrinsics is None:
                    intrin_mat[:3, :3] = torch.Tensor(cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                else:
                    intrin_mat[:3, :3] = torch.Tensor(self.tangent_intrinsics[cam])
                
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                
                # for pose network
                original_width, original_height = img.size
                """TEMP 임시로 resize_width 및 height 여기에서 정의"""
                pose_input_img = img.resize((640, 192), Image.Resampling.LANCZOS)
                pose_input_img = transforms.ToTensor()(pose_input_img).unsqueeze(0)
                
                # print(pose_input_img.shape)
                
                campose_imgs.append(pose_input_img)
                
                # for 3D object detection network
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                
                # print("resized", img.size)
                # display(img)
                
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean, self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            
            # for pose network: total cams' pose network inputs
            pose_input_imgs.append(torch.stack(campose_imgs))
            
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
                
        """ ego pose 변경 """
        if self.ego2global_rotation is None and self.ego2global_translation is None:
            # Get mean pose of all cams
            ego2global_rotation = np.mean([key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
            ego2global_translation = np.mean([key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        else:
            ego2global_rotation = self.ego2global_rotation
            ego2global_translation = self.ego2global_translation
        
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
            
            # for pose net
            pose_input_imgs
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        # print("__getitem__, idx: ", idx)
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            # print("push cam_info of key_idx ", str(key_idx))
            
            for sweep_idx in self.sweeps_idx:
                # print("push sweep_info of sweep_idx ", str(sweep_idx))
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)
        else:
            image_data_list = self.get_image(cam_infos, cams)    
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
            
            # for pose network
            pose_input_imgs
        ) = image_data_list[:8]
        
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
            
            # for pose network
            pose_input_imgs
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)

def createProjectGrid(erp_h, erp_w, tangent_h, tangent_w, num_rows, num_cols, phi_centers, fov):
    height, width = tangent_h, tangent_w

    FOV = fov
    FOV = [FOV[0] / 360.0, FOV[1] / 180.0]
    FOV = torch.tensor(FOV, dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2

    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

    num_rows = num_rows
    num_cols = num_cols
    phi_centers = phi_centers

    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []

    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols): # 0 ~ num_cols.length
            theta_interval = 360 / n_cols # 현재 row (위도)에서 쪼개질 경도 (col)의 위치
            theta_center = j * theta_interval + theta_interval / 2
            center = [theta_center, phi_centers[i]] # 각 tangent image의 center position

            # print(str(j) + " th theta center " + str(theta_center) + " phi center " + str(phi_centers[i]))
            
            all_combos.append(center)

            # 구좌표계에서의 tangent image가 차지하는 영역에 대한 좌표들
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2

            # ERP image에서 현재 tangent가 차지하는 영역에 대한 pixel 위치들
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)

            # ERP 이미지에서 현재 tangent image 영역에 해당하는 부분에 1로 마스킹
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)

    all_combos = np.vstack(all_combos)
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    n_patch = all_combos.shape[0]
    
    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)

    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2
    lon_new = lon / PI
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2

    lon_new = lon_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
    lat_new = lat_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
    
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0)

    return n_patch, grid

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# device = 'cuda:2'
# device = 'cpu'
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

""" 1. Init BEVStereo model and hyper-params """
exp_name = 'bevstereo_ema_da_key2'
ckpt_path = './ckpt/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth'
use_ema = False
data_root_dir = '../data/nuscenes/'
default_root_dir = os.path.join('./outputs/', exp_name)

gpus = 1
eval_interval = 1
batch_size_per_device = 1
basic_lr_per_img = 2e-4 / 64
mmcv.mkdir_or_exist(default_root_dir)
evaluator = DetNuscEvaluator(class_names=CLASSES, output_dir=default_root_dir)
mode = 'valid'
data_use_cbgs = False

"""
기본적으로 key_idxes [0, -1]만 사용하여 stereo matching을 하는 듯
sweep_idxes는 계속 빈 리스트로 존재하는 반면, key_idxes는 [0, -1]이 됨
코드 분석할 때 일단은 key_idxes만 보면 될 듯
"""
num_sweeps = 1
sweep_idxes = list()
key_idxes = list()

data_return_depth = True
downsample_factor = backbone_conf['downsample_factor']
dbound = backbone_conf['d_bound']
depth_channels = int((dbound[1] - dbound[0]) / dbound[2])
use_fusion = False

train_info_paths = os.path.join(data_root_dir, 'nuscenes_infos_train.pkl')
val_info_paths = os.path.join(data_root_dir, 'nuscenes_infos_val.pkl')
# predict_info_paths = os.path.join(data_root_dir, 'nuscenes_infos_test.pkl')
predict_info_paths = os.path.join(data_root_dir, 'nuscenes_infos_val.pkl')

""" BEVStereo configurations """
# base ckpt of bevstereo (2 key)
key_idxes = [-1]
head_conf['bev_backbone_conf']['in_channels'] = 80 * (len(key_idxes) + 1)
head_conf['bev_neck_conf']['in_channels'] = [80 * (len(key_idxes) + 1), 160, 320, 640]
head_conf['train_cfg']['code_weight'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
head_conf['test_cfg']['thresh_scale'] = [0.6, 0.4, 0.4, 0.7, 0.8, 0.9]
head_conf['test_cfg']['nms_type'] = 'size_aware_circle'

# ckpt with depth aggregation
backbone_conf['use_da'] = True
data_use_cbgs = True
basic_lr_per_img = 2e-4 / 32

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

model = BEVStereo(backbone_conf, head_conf, is_train_depth=True)

checkpoint = torch.load(ckpt_path)
model_weights = checkpoint["state_dict"]

for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)

""" 2. Init PoseNet """
manydepth_model_path = './manydepth/ckpt/KITTI_MR/'

# pose_enc_dict = torch.load(os.path.join(manydepth_model_path, "pose_encoder.pth"), map_location=device)
# pose_dec_dict = torch.load(os.path.join(manydepth_model_path, "pose.pth"), map_location=device)

pose_enc_dict = torch.load(os.path.join(manydepth_model_path, "pose_encoder.pth"))
pose_dec_dict = torch.load(os.path.join(manydepth_model_path, "pose.pth"))

pose_enc = ResnetEncoder(18, False, num_input_images=2)
pose_dec = PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

pose_enc.load_state_dict(pose_enc_dict, strict=True)
pose_dec.load_state_dict(pose_dec_dict, strict=True)

model.eval()
model.cuda()
pose_enc.eval()
pose_dec.eval()
pose_enc.cuda()
pose_dec.cuda()
print("BEVStereo and PoseNet loaded")

""" 3. Init Tangent Projection Grid """
num_rows = 1
num_cols = [6]
phi_centers = [0]

# fov 가로/세로 비율과 nuscenes input의 width/height 비율이 같음
# 900/1600, 396/704: 1.777777...
# 위 비율에 맞춰서 tangent patch size 결정하기
# 중요! 704, 256은 aspect ratio가 다름. 원래 코드에서도 704, 396으로 resize하고 추후에 704, 256으로 crop함
tangent_h = 396 # 256 # 900 # 396
tangent_w = 704 # 704 # 1600 # 704
fov  = [70, 39.375]
erp_h, erp_w = 1920, 3840

n_patch, grid = createProjectGrid(erp_h, erp_w, tangent_h, tangent_w, num_rows, num_cols, phi_centers, fov)
grid = grid.cuda()

vis_tangent_h = 900 # visualization resolution
vis_tangent_w = 1600

n_patch, vis_grid = createProjectGrid(erp_h, erp_w, vis_tangent_h, vis_tangent_w, num_rows, num_cols, phi_centers, fov)








""" Init Insta360 ERP Dataset Meta-data """

# scene_dir = "scene_2/"
erp_img_root = "../data/daejeon_road_outdoor/erp_images/"
# save_path = './outputs/temp/' + scene_dir
save_path = './outputs/det_result_imgs/' + scene_dir

erp_imgs = []
for filename in os.listdir(erp_img_root + scene_dir):
    erp_imgs.append(os.path.join(erp_img_root, scene_dir, filename))

""" Input image configuration """ 
ida_aug_conf['cams'] = ['CAM_FRONT_LEFT',
                        'CAM_FRONT',
                        'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT',
                        'CAM_BACK',
                        'CAM_BACK_RIGHT']

H, W = ida_aug_conf['H'], ida_aug_conf['W']
fH, fW = ida_aug_conf['final_dim']
resize = max(fH / H, fW / W)
resize_dims = (int(W * resize), int(H * resize))
newW, newH = resize_dims
crop_h = int((1 - np.mean(ida_aug_conf['bot_pct_lim'])) * newH) - fH # 396 (newH) - 256 (fH) = 140
crop_w = int(max(0, newW - fW) / 2) # max(0, 704 - 704) = 0
crop = (crop_w, crop_h, crop_w + fW, crop_h + fH) # (0, 140, 704, 396)

tangent_intrinsics = {'CAM_FRONT_LEFT': [[1.31669199e+03, 0.00000000e+00, 7.71567974e+02], # Tangent location 0 (leftmost)
                                         [0.00000000e+00, 1.30594375e+03, 4.27529182e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                      'CAM_FRONT': [[1.32277551e+03, 0.00000000e+00, 7.56801337e+02], # Tangent location 1
                                    [0.00000000e+00, 1.31076362e+03, 4.17111552e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                      'CAM_FRONT_RIGHT': [[1.31303854e+03, 0.00000000e+00, 7.16879740e+02], # Tangent location 2
                                          [0.00000000e+00, 1.30008012e+03, 4.21897818e+02],
                                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                      'CAM_BACK_LEFT': [[1.31538668e+03, 0.00000000e+00, 7.62655552e+02], # Tangent location 3
                                        [0.00000000e+00, 1.30582175e+03, 4.21392564e+02],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                      'CAM_BACK': [[1.37645753e+03, 0.00000000e+00, 7.33078005e+02], # Tangent location 4
                                   [0.00000000e+00, 1.36126888e+03, 4.13681559e+02],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                      'CAM_BACK_RIGHT': [[1.33525055e+03, 0.00000000e+00, 7.02715248e+02], # Tangent location 5
                                         [0.00000000e+00, 1.32071237e+03, 4.12790092e+02],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]}

"""sensor2ego (calibrated_sensor) rotation degree를 insta360에 맞게 변경"""
# BEVFormer에서 사용한 sensor2lidar [roll, pitch, yaw]
# sensor2ego_rot_eulers = {'CAM_FRONT': [-88.89452145820958, -0.34991764317283675, 0.0],
#                          'CAM_FRONT_RIGHT': [-89.73833720738743, 1.5337073678165962, -60.0],
#                          'CAM_FRONT_LEFT': [-89.2717432442176, -1.221029004801636, 60.0],
#                          'CAM_BACK': [-90.4448247670561, 0.552239989680819, 180.0],
#                          'CAM_BACK_LEFT': [-91.67728722938543, -1.4284019448799024, 120.0],
#                          'CAM_BACK_RIGHT': [-91.13862482181912, 2.0544464501438036, -120.0]}

# BEVFormer에서 사용한 sensor2ego
# sensor2ego_rot_eulers = {'CAM_FRONT': [-90.32322642770005, -0.046127194838589326, -90.0],
#                          'CAM_FRONT_RIGHT': [-90.7820235885, 0.5188438566959973, -150.0],
#                          'CAM_FRONT_LEFT': [-89.85977500319999, 0.12143609391200436, -30.0],
#                          'CAM_BACK': [-89.0405962694, 0.22919685786400154, 90.0],
#                          'CAM_BACK_LEFT': [-90.91736319750001, -0.21518275753700122, 30.0],
#                          'CAM_BACK_RIGHT': [-90.93206677999999, 0.6190947610589966, -210.0]}

# sensor2ego_rot_eulers = {'CAM_FRONT_LEFT': [-90.0, 0.0, 60.0],
#                          'CAM_FRONT': [-90.0, 0.0, 0.0],
#                          'CAM_FRONT_RIGHT': [-90.0, 0.0, -60.0],
#                          'CAM_BACK_LEFT': [-90.0, 0.0, 120.0],
#                          'CAM_BACK': [-90.0, 0.0, 180.0],
#                          'CAM_BACK_RIGHT': [-90.0, 0.0, -120.0]}

sensor2ego_rot_eulers = {'CAM_FRONT_LEFT': [-90.0, 0.0, 60.0],
                         'CAM_FRONT': [-90.0, 0.0, 0.0],
                         'CAM_FRONT_RIGHT': [-90.0, 0.0, -60.0],
                         'CAM_BACK_LEFT': [-90.0, 0.0, 240.0],
                         'CAM_BACK': [-90.0, 0.0, 180.0],
                         'CAM_BACK_RIGHT': [-90.0, 0.0, 120.0]}

# sensor2ego_rot_eulers = {'CAM_FRONT_LEFT': [-90.0, 0.0, -150.0],
#                          'CAM_FRONT': [-90.0, 0.0, -90.0],
#                          'CAM_FRONT_RIGHT': [-90.0, 0.0, -30.0],
#                          'CAM_BACK_LEFT': [-90.0, 0.0, 30.0],
#                          'CAM_BACK': [-90.0, 0.0, 90.0],
#                          'CAM_BACK_RIGHT': [-90.0, 0.0, 150.0]}

# sensor2ego_rots = [] # tcam2egocam_rots
# for i in range(len(ida_aug_conf['cams'])):
#     cam = ida_aug_conf['cams'][i]
#     sensor2ego_degrees = sensor2ego_euler_degrees[cam]
#     sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
#     sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
#     sensor2ego_r_mat = sensor2ego_q.rotation_matrix
#     sensor2ego_rots.append(sensor2ego_r_mat)

#     # 확인
#     # radians = euler_from_quaternion(sensor2ego_q)
#     # print(cam, sensor2ego_q)
#     # print(cam, [euler / np.pi * 180 for euler in radians])

# x, y 좌표는 ego sensor와 virtual tangent sensor 간에 차이가 없는게 맞음
# 그럼 왜 z는?
# sensor2ego_trans = [0.0, 0.0, 0.0]
sensor2ego_trans = [0.0, 0.0, 1.5]

ego2global_rotation = np.array([1.0, 0.0, 0.0, 0.0])
ego2global_translation = np.array([0.0, 0.0, 0.0])

""" Init Nusc Dataset """
dataset = NuscDetDataset(ida_aug_conf=ida_aug_conf,
                         bda_aug_conf=bda_aug_conf,
                         classes=CLASSES,
                         data_root=data_root_dir,
                         info_paths=predict_info_paths,
                         is_train=False,
                         
                         # Dataset customization
                         tangent_intrinsics=tangent_intrinsics,
                         sensor2ego_rot_eulers=sensor2ego_rot_eulers,
                         sensor2ego_trans=sensor2ego_trans,
                         ego2global_rotation=ego2global_rotation,
                         ego2global_translation=ego2global_translation,
                         
                         infos=infos,
                         
                         img_conf=img_conf,
                         num_sweeps=num_sweeps,
                         sweep_idxes=sweep_idxes,
                         key_idxes=key_idxes,
                         return_depth=use_fusion,
                         use_fusion=use_fusion)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size_per_device,
    shuffle=False,
    collate_fn=partial(collate_fn, is_return_depth=use_fusion),
    num_workers=4,
    sampler=None,
)

data_iterator = iter(data_loader)
data = next(data_iterator)

for idx in range(len(erp_imgs)):
    
    (_, mats, _, img_metas, _, _, campose_imgs) = data

    for key, value in mats.items():
        mats[key] = value.cuda()

    # Collect adjacent erp images
    adj_erp_imgs = list()
    for key_idx in [0, -1]: # current idx first
        cur_idx = key_idx + idx
        if cur_idx < 0: # first frame of the scene video
            cur_idx = idx

        # load images
        fname = erp_imgs[cur_idx]
        print(f"Load {cur_idx}th erp image")
        erp_img = cv2.imread(fname, cv2.IMREAD_COLOR)
        if erp_img is None:
            continue
        erp_img = erp_img.astype(np.float32) / 255
        erp_img = np.transpose(erp_img, [2, 0, 1]) # permutation, 세 번째 axis가 첫 번째 axis로
        erp_img = torch.from_numpy(erp_img) # Create Tensor from numpy array
        erp_img = erp_img.unsqueeze(0) # Increase Tensor dimension by 1

        adj_erp_imgs.append(erp_img)

        
    # 시작!
    with torch.no_grad():
        starter.record()

        # to GPU memory (시간 측정?)
        for i, erp_img in enumerate(adj_erp_imgs):
            adj_erp_imgs[i] = erp_img.cuda()

        # Tangent projection with GPU
        persp_seqs = list()
        for erp_img in adj_erp_imgs:
            persp = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            persp_reshape = F.unfold(persp, kernel_size=(tangent_h, tangent_w), stride=(tangent_h, tangent_w))
            persp_reshape = persp_reshape.reshape(1, 3, tangent_h, tangent_w, n_patch)
            persp_seqs.append(persp_reshape)

        # Re-shape patches and prepare camera pose estimation inputs
        sweep_patches = list()
        sweep_pose_inputs = list()
        for persp_seq in persp_seqs:
            patches = list()
            campose_inputs = list()
            for i in range(len(ida_aug_conf['cams'])):
                patch = persp_seq[0, :, :, :, i]

                # Resize patch for camera pose input
                # campose_input = F.interpolate(patch.unsqueeze(0), size=(360, 640), # Image resize with interpolate function
                #                               mode='bicubic', align_corners=False) # bicubic, bilinear, ...
                # # campose_input = transforms.ToPILImage()(campose_input[0])
                # # display(campose_input)
                # campose_inputs.append(campose_input)

                # Color change (확인) - (2023-1-29) 결과가 더 안 좋은듯?
                patch = patch.permute(1, 2, 0).cpu().numpy()
                patch = patch * 255
                patch = patch[:,:,[2,1,0]].astype(np.uint8)

                # image adjustment and normalization for model input
                patch = transforms.ToPILImage()(patch) # time (ms):  5.7   

                campose_input_crop = (0, 168, 640, 360)
                campose_input = patch.crop(campose_input_crop)
                # display(campose_input)
                campose_input = transforms.ToTensor()(campose_input).unsqueeze(0)
                # print(campose_input.shape)
                campose_input = campose_input.cuda()
                campose_inputs.append(campose_input)

                patch = patch.crop(crop)
                patch = mmcv.imnormalize(np.array(patch), 
                                         np.array(img_conf['img_mean'], np.float32), # TODO check: img_mean, img_std?
                                         np.array(img_conf['img_std'], np.float32),
                                         img_conf['to_rgb']) # time (ms):  5.79
                patch = torch.from_numpy(patch).permute(2, 0, 1) # time (ms):  0.34
                patch = patch.cuda() # time (ms):  1.20

                patches.append(patch)
            sweep_patches.append(torch.stack(patches, 0))
            sweep_pose_inputs.append(torch.stack(campose_inputs))
        sweep_patches = torch.stack(sweep_patches, 0).unsqueeze(0)

        # Camera pose estimation
        campose_mats = list()
        for cam_idx in range(len(ida_aug_conf['cams'])):
            source_image = sweep_pose_inputs[0][cam_idx]
            target_image = sweep_pose_inputs[1][cam_idx]
            pose_inputs = [source_image, target_image]
            # if cam_idx == 0:
            #     print(torch.cat(pose_inputs, 1).shape)
            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = pose_dec(pose_inputs)
            # print("CAM", ida_aug_conf['cams'][cam_idx])
            # print("pose axis angle", axisangle)
            # print("pose translation", translation)
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
            # print(f"Pose matrix for image pair {cam_idx}: \n{pose}")
            campose_mats.append(pose)

        sensor2sensor_mats = list()
        sensor2sensor_mats.append(torch.stack(campose_mats, 0).unsqueeze(0))
        sensor2sensor_mats.append(sensor2sensor_mats[0].inverse())

        # 3D object detection
        preds = model(sweep_patches, mats, posenet_outputs=sensor2sensor_mats)

        
    # CenterPoint 모델의 get_bboxes 함수로, img_metas는 img_metas[i]['box_type_3d']로 밖에 안쓰임
    # 즉, ego pose 값은 쓰이지 않음. 그저 예측 결과들에서 NMS 수행하고 box, score, label 내놓음
    results = model.get_bboxes(preds, img_metas)

    for i in range(len(results)):
        results[i][0] = results[i][0].detach().cpu().numpy()
        results[i][1] = results[i][1].detach().cpu().numpy()
        results[i][2] = results[i][2].detach().cpu().numpy()
        results[i].append(img_metas[i])

    """ format bbox results """
    pred_results = results[0][:3]
    boxes, scores, labels = pred_results

    img_metas = results[0][3]
    sample_token = img_metas['token']
    trans = np.array(img_metas['ego2global_translation'])
    rot = Quaternion(img_metas['ego2global_rotation'])

    nusc_annos = {}
    annos = list()
    for i, box in enumerate(boxes):
        name = CLASSES[labels[i]]
        center = box[:3]
        wlh = box[[4, 3, 5]]
        box_yaw = box[6]
        box_vel = box[7:].tolist()
        box_vel.append(0)
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
        nusc_box = Box(center, wlh, quat, velocity=box_vel)

        nusc_box.rotate(rot)
        nusc_box.translate(trans)

        if np.sqrt(nusc_box.velocity[0]**2 + nusc_box.velocity[1]**2) > 0.2:
            if name in ['car','construction_vehicle','bus','truck','trailer']:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = DefaultAttribute[name]
        nusc_anno = dict(
            sample_token=sample_token,
            translation=nusc_box.center.tolist(),
            size=nusc_box.wlh.tolist(),
            rotation=nusc_box.orientation.elements.tolist(),
            velocity=nusc_box.velocity[:2],
            detection_name=name,
            detection_score=float(scores[i]),
            attribute_name=attr,
        )
        annos.append(nusc_anno)
    nusc_annos[sample_token] = annos

    modality=dict(use_lidar=False,
                  use_camera=True,
                  use_radar=False,
                  use_map=False,
                  use_external=False)
    nusc_submissions = {
        'meta': modality,
        'results': nusc_annos,
    }

    """ save results to json file"""
    jsonfile_prefix = os.path.dirname('./outputs/' + exp_name + '/')
    mmcv.mkdir_or_exist(jsonfile_prefix)
    res_path = os.path.join(jsonfile_prefix, 'results_nusc.json')
    mmcv.dump(nusc_submissions, res_path)

    """ 4. Init projection grid and ERP image for VIS """
    fname = erp_imgs[idx]
    vis_erp_img = cv2.imread(fname, cv2.IMREAD_COLOR)
    vis_erp_img = vis_erp_img.astype(np.float32) / 255
    vis_erp_img = np.transpose(vis_erp_img, [2, 0, 1]) # permutation, 세 번째 axis가 첫 번째 axis로
    vis_erp_img = torch.from_numpy(vis_erp_img) # Create Tensor from numpy array
    vis_erp_img = vis_erp_img.unsqueeze(0) # Increase Tensor dimension by 1

    vis_grid = vis_grid.cuda()
    vis_erp_img = vis_erp_img.cuda()

    vis_persp = F.grid_sample(vis_erp_img, vis_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    vis_persp_reshape = F.unfold(vis_persp, kernel_size=(vis_tangent_h, vis_tangent_w), stride=(vis_tangent_h, vis_tangent_w))
    vis_persp_reshape = vis_persp_reshape.reshape(1, 3, vis_tangent_h, vis_tangent_w, n_patch).cpu()

    result_path = './outputs/bevstereo_ema_da_key2/results_nusc.json'
    data_root = '../data/nuscenes/'
    results = mmcv.load(result_path)['results']
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ]

    # IMG_KEYS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

    # Get data from dataset
    info = infos[0] # nusc dataset의 idx는 0으로 고정

    ego2global_rotation = img_metas['ego2global_rotation']
    ego2global_translation = img_metas['ego2global_translation']

    # Set cameras
    threshold = 0.4
    show_range = 60

    # Set figure size
    plt.figure(figsize=(21, 8))

    imsize = (1600, 900)
    box_vis_level = BoxVisibility.ANY

    for i, k in enumerate(ida_aug_conf['cams']):
        # Draw camera views
        fig_idx = i + 1 if i < 3 else i + 1
        plt.subplot(2, 3, fig_idx)

        # Set camera attributes
        plt.title(k)
        plt.axis('off')
        plt.xlim(0, 1600)
        plt.ylim(900, 0)
        # plt.ylim(900, 327)

        cur_patch = vis_persp_reshape[0, :, :, :, i].permute(1, 2, 0).numpy()
        cur_patch = cur_patch * 255
        img = cur_patch[:,:,[2,1,0]].astype(np.uint8)

        # vis_crop = (0, 327, 1600, 900)
        # img = transforms.ToPILImage()(img) # time (ms):  5.7
        # img = img.crop(vis_crop)

        # Draw images
        plt.imshow(img)

        if dataset.sensor2ego_trans is None:
            sensor2ego_trans = info['cam_infos'][k]['calibrated_sensor']['translation']
        else:
            sensor2ego_trans = dataset.sensor2ego_trans

        if dataset.sensor2ego_rot_eulers is None:
            sensor2ego_rot = info['cam_infos'][k]['calibrated_sensor']['rotation']
        else:
            sensor2ego_degrees = dataset.sensor2ego_rot_eulers[k]
            sensor2ego_radians = [degree * np.pi / 180 for degree in sensor2ego_degrees]
            # print(sensor2ego_degrees)
            sensor2ego_q = Quaternion(get_quaternion_from_euler(sensor2ego_radians))
            sensor2ego_rot = sensor2ego_q

        if dataset.tangent_intrinsics is None:
            intrinsic = info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic']
        else:
            intrinsic = tangent_intrinsics[k]
        intrinsic = np.array(intrinsic)

        # print(k)
        # print(sensor2ego_trans)
        # print(sensor2ego_rot)
        # print(intrinsic)

        boxes_pred = []
        for box_dict in results[info['sample_token']]:
            if box_dict['detection_score'] >= threshold and box_dict['detection_name'] in show_classes:
                box = Box(
                    box_dict['translation'],
                    box_dict['size'],
                    Quaternion(box_dict['rotation']),
                    name=box_dict['detection_name']
                )

                # Calculate box coordinate in the ego coord. system
                trans = -np.array(ego2global_translation)
                rot = Quaternion(ego2global_rotation).inverse
                box.translate(trans)
                box.rotate(rot)
                boxes_pred.append(box)

        # box를 ego => sensor로 이동
        for box in boxes_pred:
            box.translate(-np.array(sensor2ego_trans))
            box.rotate(Quaternion(sensor2ego_rot).inverse)
            # cam_corners = cam_corners @ Quaternion(sensor2ego_rot).inverse.rotation_matrix.T

            if box_in_image(box, intrinsic, imsize, vis_level=box_vis_level):
                c=cm.get_cmap('tab10')(show_classes.index(box.name))
                box.render(plt, view=intrinsic, normalize=True, colors=(c, c, c))

    # Set legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(),
               by_label.keys(),
               loc='upper right',
               framealpha=1)

    plt.tight_layout(w_pad=0, h_pad=2)
    save_name ='output_%04d.jpg' % idx
    mmcv.mkdir_or_exist(save_path)
    plt.savefig(save_path+save_name)
    # plt.show()

    plt.close()