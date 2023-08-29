import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from src.models.losses import DiceLoss, CEDiceLoss, FocalLoss
from src.models.transformer.layers import AttentionLayers


class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)
        x = Ms * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# MSRAM
class MultiResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.conv0 = nn.Conv3d(chan, chan, 3, padding=1)  #
        self.conv1 = nn.Conv3d(chan, chan, 3, padding=1)
        self.conv2 = nn.Conv3d(chan, chan, 5, padding=2)
        self.conv4 = nn.Conv3d(2 * chan, chan, 1)
        self.conv5 = nn.Conv3d(2 * chan, chan, 1)
        self.conv6 = nn.Conv3d(2 * chan, chan, 1)
        self.conv8 = nn.Conv3d(chan, chan, 3, padding=1)
        self.conv9 = nn.Conv3d(chan, chan, 5, padding=2)
        self.conv11 = nn.Conv3d(chan, chan, 3, padding=1)

        self.act = nn.SiLU()
        self.ResBlock1 = Res3Block(chan)
        self.ResBlock2 = Res3Block(chan)
        self.ResBlock3 = Res5Block(chan)
        self.ResBlock4 = Res5Block(chan)
        self.ca = ChannelAttention(2 * chan, ratio=chan // 4)

    def forward(self, x):
        f = self.act(self.conv0(x))

        res1 = self.conv1(f)
        res11 = self.ResBlock1(res1)
        res111 = self.ResBlock2(res11)
        res1111 = self.conv5(torch.cat((res11, res111), dim=1))
        res1_o = self.conv8(res1111) + res1

        res2 = self.conv2(f)
        res22 = self.ResBlock3(res2)
        res222 = self.ResBlock4(res22)
        res2222 = self.conv6(torch.cat((res22, res222), dim=1))
        res2_o = self.conv9(res2222) + res2

        res = self.conv4(self.ca(torch.cat((res1_o, res2_o), dim=1)))
        res = self.conv11(res) + x
        return res

# RDAB scale = 3
class Res3Block(nn.Module):
    def __init__(self, chan):
        super(Res3Block, self).__init__()
        self.conv1 = nn.Conv3d(chan, chan, 3, padding=1)
        self.conv2 = nn.Conv3d(chan, chan, 3, padding=1)
        # self.conv3 = nn.Conv3d(chan, chan, 3, padding=1)
        self.conv4 = nn.Conv3d(3 * chan, chan, 1)
        self.sa = SpatialAttention(chan)
        self.act = nn.SiLU()
        # self.drop = nn.Dropout3d(0.1)


    def forward(self, x):
        res1 = self.act(self.conv1(x))
        res2 = self.act(self.conv2(res1 + x))
        res3 = self.conv4(torch.cat((x, res1, res2), dim=1))
        # x = self.act(self.conv3(x))
        res = self.sa(x + res3)
        return res

# RDAB scale = 5
class Res5Block(nn.Module):
    def __init__(self, chan):
        super(Res5Block, self).__init__()
        self.conv1 = nn.Conv3d(chan, chan, 5, padding=2)
        self.conv2 = nn.Conv3d(chan, chan, 5, padding=2)
        # self.conv3 = nn.Conv3d(chan, chan, 5, padding=2)
        self.conv4 = nn.Conv3d(3 * chan, chan, 1)
        self.sa = SpatialAttention(chan)
        self.act = nn.SiLU()
        # self.drop = nn.Dropout3d(0.1)

    def forward(self, x):
        res1 = self.act(self.conv1(x))
        res2 = self.act(self.conv2(res1 + x))
        res3 = self.conv4(torch.cat((x, res1, res2), dim=1))
        # x = self.act(self.conv3(x))
        res = self.sa(x + res3)
        return res


class VoxelDecoderMLP(nn.Module):
    def __init__(
            self,
            patch_num: int = 4,
            voxel_size: int = 32,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )

        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def generate(
            self,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(
            self,
            context: Tensor,
            context_mask: Tensor = None
    ) -> Tensor:
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = out.view(self.patch_num ** 3 * context.shape[0], -1)
        patched = self.to_patch(out)

        return patched.view(-1, self.voxel_size, self.voxel_size, self.voxel_size)

    def get_loss(
            self,
            x: Tensor,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return F.binary_cross_entropy_with_logits(
            out.view(out.size(0), -1),
            x.view(out.size(0), -1)
        )


class VoxelDecoderCNN(nn.Module):
    def __init__(
            self,
            patch_num: int = 4,
            num_cnn_layers: int = 3,
            num_multires_blocks: int = 2,
            cnn_hidden_dim: int = 64,
            voxel_size: int = 32,
            dim: int = 512,
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )

        has_multiresblocks = num_multires_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = dim if not has_multiresblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        self.conv1 = nn.Conv3d(dim, dec_chans[1], 1)
        self.MSRAM = MultiResBlock(dec_chans[1])
        sub_dec_layers = []
        for (dec_in, dec_out) in dec_chans_io:
            sub_dec_layers.append(
                nn.Sequential(nn.ConvTranspose3d(dec_in, dec_out, 4, stride=2, padding=1), nn.SiLU()))

        sub_dec_layers.append(nn.Conv3d(dec_chans[-1], 1, 1))
        self.VRB = nn.Sequential(*sub_dec_layers)
        self.act = nn.SiLU()

        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def generate(
            self,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(
            self,
            context: Tensor,
            context_mask: Tensor = None
    ) -> Tensor:
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = rearrange(out, 'b (h w c) d -> b d h w c', h=self.patch_num, w=self.patch_num, c=self.patch_num)
        out = self.act(self.conv1(out))
        res = self.MSRAM(out)
        res = self.VRB(res)
        return res

    def get_loss(
            self,
            x: Tensor,
            context: Tensor,
            context_mask: Tensor = None,
            loss_type='dice'
    ):
        out = self(context, context_mask)
        out = out.view(out.size(0), -1)
        x = x.view(out.size(0), -1)

        if loss_type == 'ce':
            loss_fn = F.binary_cross_entropy_with_logits
        elif loss_type == 'dice':
            loss_fn = DiceLoss()
        elif loss_type == 'ce_dice':
            loss_fn = CEDiceLoss()
        elif loss_type == 'focal':
            loss_fn = FocalLoss()
        else:
            raise ValueError(f'Unsupported loss type "{loss_type}"')

        return loss_fn(out, x)


