import torch
import torch.nn as nn
import math
from .LWGA import MultiScaleAttentionBlock_Modified
from .phase_classifier import PhaseClassifier

# Embedding and sampling modules adapted from Restormer; see THIRD_PARTY_NOTICES.md.


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class TLP(nn.Module):
    def __init__(self, inp_A_channels=1, out_channels=1,
                 dim=48, num_blocks=(1, 2, 2),
                 num_refinement_blocks=2,
                 bias=False,
                 num_classes=2,
                 att_kernel=11
                 ):
        super(TLP, self).__init__()

        self.phase_classifier = PhaseClassifier(in_channels=out_channels, num_classes=num_classes)

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, att_kernel=att_kernel)

        self.attention_spatial = Attention_spatial(dim * 2 ** 2)

        self.phase_guidance_3 = FeatureWiseAffine(in_channels=num_classes, out_channels=dim * 2 ** 2)

        self.decoder_level3 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 2), att_kernel=att_kernel) for i in range(num_blocks[2])])

        self.phase_guidance_2 = FeatureWiseAffine(in_channels=num_classes, out_channels=dim * 2 ** 1)
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 1), att_kernel=att_kernel) for i in range(num_blocks[1])])

        self.phase_guidance_1 = FeatureWiseAffine(in_channels=num_classes, out_channels=dim)
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 1), att_kernel=att_kernel) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 1), att_kernel=att_kernel) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, c_pre, c_a=None, phase_index=None):
        """Select a branch from the target during training or use a zero-based branch index for inference."""
        if c_pre.ndim != 4 or any(size % 4 for size in c_pre.shape[-2:]):
            raise ValueError("输入必须为 [B, C, H, W]，且高宽均可被 4 整除")
        if phase_index is not None:
            if not isinstance(phase_index, int) or not 0 <= phase_index < self.phase_classifier.num_classes:
                raise ValueError("phase_index 超出有效的分支编号范围")
            phase = c_pre.new_zeros((c_pre.shape[0], self.phase_classifier.num_classes))
            phase[:, phase_index] = 1
        elif c_a is not None:
            phase = self.phase_classifier(c_a)
        else:
            raise ValueError("训练需提供目标图像 c_a；推理需指定 phase_index")
        out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(c_pre)
        out_enc_level3 = self.attention_spatial(out_enc_level3_A)
        out_enc_level3 = self.phase_guidance_3(out_enc_level3, phase)

        out_dec_level3 = self.decoder_level3(out_enc_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.phase_guidance_2(inp_dec_level2, phase)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2_A], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.phase_guidance_1(inp_dec_level1, phase)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_A], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        out_dec_level1 = self.tanh(out_dec_level1)

        return out_dec_level1


class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class Encoder_A(nn.Module):
    def __init__(self, inp_channels=2, dim=48, num_blocks=(1, 2, 2), att_kernel=9):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=dim, att_kernel=att_kernel) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 1), att_kernel=att_kernel) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            MultiScaleAttentionBlock_Modified(dim=int(dim * 2 ** 2), att_kernel=att_kernel) for i in range(num_blocks[2])])

    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)
        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)
        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        return out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


# Embed overlapping image patches with a 3x3 convolution.
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# Spatial sampling modules.
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, phase):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(phase).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Build a convolutional downsampling block for the discriminator."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),

            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
