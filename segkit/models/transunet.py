from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3):
        pad = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.skip_proj = None if skip_ch in (0, None) else nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + (0 if self.skip_proj is None else out_ch), out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class TransUNet(nn.Module):
    """ViT-hybrid + ResNet-50 U-Net style segmenter.

    Architecture is unchanged from the original repo; fixes are limited to
    bookkeeping (no aliased sub-modules, skip channels read from the backbone,
    ResNet pretrained flag honoured).
    """

    def __init__(
        self,
        vit_name="vit_base_r50_s16_224.orig_in21k",
        pretrained=True,
        num_classes=1,
        decoder_channels=(256, 128, 64, 16),
        skip_channels=None,
        in_chans=1,
        resnet_name="resnet50",
        resnet_pretrained=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decoder_channels = list(decoder_channels)
        if resnet_pretrained is None:
            resnet_pretrained = pretrained

        # --- ViT (hybrid) branch
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=224,
            in_chans=3,
        )
        self.input_adapter = nn.Conv2d(in_chans, 3, kernel_size=1, bias=False)
        self.vit_projection = nn.Conv2d(self.vit.num_features, 512, kernel_size=1, bias=False)

        # --- ResNet branch (multi-scale skips)
        self.resnet = timm.create_model(
            resnet_name, pretrained=resnet_pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        res_ch = list(self.resnet.feature_info.channels())  # e.g. [256, 512, 1024, 2048]
        if len(res_ch) != 4:
            raise RuntimeError(f"Expected 4 ResNet feature maps, got channels {res_ch}")
        self.latent = nn.Sequential(
            ConvBNReLU(res_ch[3], 1024),
            ConvBNReLU(1024, 512),
        )

        # Skip channels of decoder stages 0..3 are [res stage3, res stage2, res stage1, none].
        self.decoder_skip_channels = [res_ch[2], res_ch[1], res_ch[0], 0]
        if skip_channels is not None and list(skip_channels) != self.decoder_skip_channels:
            warnings.warn(
                f"model.skip_channels={list(skip_channels)} in the config does not match the real "
                f"ResNet skip channels {self.decoder_skip_channels}; using the real ones."
            )
        self.skip_channels = list(self.decoder_skip_channels)

        self.decoder = nn.ModuleList()
        in_ch = 512
        for out_ch, skip_ch in zip(self.decoder_channels, self.decoder_skip_channels):
            self.decoder.append(DecoderBlock(in_ch, out_ch, skip_ch=skip_ch))
            in_ch = out_ch

        self.head = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)

    def _encode_vit(self, x):
        vit = self.vit
        x = vit.patch_embed(x)
        batch_size, num_patches, channels = x.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected a square ViT patch grid, got {num_patches} tokens.")

        prefix_tokens = int(vit.num_prefix_tokens)
        position_prefix_tokens = 0 if vit.no_embed_class else prefix_tokens
        old_grid_h, old_grid_w = vit.patch_embed.grid_size
        patch_pos_embed = vit.pos_embed[:, position_prefix_tokens:, :]
        if patch_pos_embed.shape[1] != old_grid_h * old_grid_w:
            raise RuntimeError("ViT positional embedding does not match the configured patch grid.")
        patch_pos_embed = patch_pos_embed.reshape(1, old_grid_h, old_grid_w, channels)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(grid_size, grid_size),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, channels)

        prefix = []
        if vit.cls_token is not None:
            prefix.append(vit.cls_token.expand(batch_size, -1, -1))
        if getattr(vit, "reg_token", None) is not None:
            prefix.append(vit.reg_token.expand(batch_size, -1, -1))

        if vit.no_embed_class:
            x = x + patch_pos_embed
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
        else:
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
            x = x + torch.cat([vit.pos_embed[:, :prefix_tokens], patch_pos_embed], dim=1)

        x = vit.blocks(x)
        x = vit.norm(x)
        x = x[:, prefix_tokens:, :]
        return x.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)

    def forward(self, x):
        x_rgb = self.input_adapter(x)
        resnet_features = self.resnet(x_rgb)
        if len(resnet_features) < 4:
            raise RuntimeError("Expected ResNet feature maps at 4 stages.")
        skip1, skip2, skip3, skip4 = resnet_features[-4:]

        vit_feat = self._encode_vit(x_rgb)
        vit_feat = F.interpolate(vit_feat, size=skip4.shape[-2:], mode="bilinear", align_corners=False)
        vit_feat = self.vit_projection(vit_feat)
        latent = self.latent(skip4) + vit_feat
        x = latent
        decoder_inputs = [skip3, skip2, skip1, None]
        for idx, block in enumerate(self.decoder):
            skip = decoder_inputs[idx] if idx < len(decoder_inputs) else None
            x = block(x, skip)
        logits = self.head(x)
        return F.interpolate(logits, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
