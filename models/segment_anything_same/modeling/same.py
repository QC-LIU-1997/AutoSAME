# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Optional, Tuple, Type, Dict, List

from .image_encoder import ImageEncoderViT,ImageEncoderViTPP, qkvAttention
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .common import MLPBlock
from einops import rearrange



class APGAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.scale = dim**-0.5

        self.q= nn.Linear(dim, dim, bias=qkv_bias)
        self.k= nn.Linear(dim, dim, bias=qkv_bias)
        self.v= nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v:torch.Tensor) -> torch.Tensor:
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)


        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = self.proj(x)

        return x

class AutoPromptGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,       
        task_num: int = 5,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.task_num = task_num
        self.cross_att1 = APGAttention(embed_dim)
        self.cross_att2 = APGAttention(embed_dim)
        self.cross_att3 = APGAttention(embed_dim)
        self.cross_att4 = APGAttention(embed_dim)

        self.cross_attf1 = APGAttention(embed_dim)
        self.cross_attf2 = APGAttention(embed_dim)
        self.cross_attf3 = APGAttention(embed_dim)
        self.cross_attf4 = APGAttention(embed_dim)

        self.mlp1 = MLPBlock(embedding_dim=embed_dim, mlp_dim=embed_dim, act=act_layer)
        self.mlp2 = MLPBlock(embedding_dim=embed_dim, mlp_dim=embed_dim, act=act_layer)
        self.mlp3 = MLPBlock(embedding_dim=embed_dim, mlp_dim=embed_dim, act=act_layer)
        self.mlp4 = MLPBlock(embedding_dim=embed_dim, mlp_dim=embed_dim, act=act_layer)

        self.task_token = nn.Embedding(task_num, embed_dim)
        self.mask_adapter = nn.Sequential(nn.Conv2d(self.embed_dim,self.embed_dim//4,1),
                                          nn.Conv2d(self.embed_dim//4,self.embed_dim//4,1),
                                          nn.Conv2d(self.embed_dim//4,self.embed_dim//4,1),
                                          nn.Conv2d(self.embed_dim//4,self.embed_dim,1),)


    def forward(
        self,
        imge,
        maske,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates (b N_points 2)
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed (b 4)
          masks (torch.Tensor or none): masks to embed (b 1 h w)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = imge.shape[0]

        # print(maske.shape,self.task_token.weight.shape,imge.shape)
        cross_1 = self.cross_att1(self.task_token.weight,maske,maske)
        cross_2 = self.cross_att2(maske,self.task_token.weight,self.task_token.weight)
        cross_f1 = self.cross_attf1(cross_1,cross_2,cross_2)
        cross_f2 = self.cross_attf2(cross_2,cross_1,cross_1)
        temp_token_t = self.mlp1(cross_f1)+self.task_token.weight
        temp_token_m = self.mlp2(cross_f2)+maske
        temp_token_concat = torch.concat([temp_token_t,temp_token_m], dim=0)
        # print(temp_token_concat.shape)

        B,C,H,W = imge.shape
        imge = imge.reshape(B,C,H*W).permute(0,2,1)
        # print(imge.shape,temp_token_concat.shape)
        cross_3 = self.cross_att3(imge,temp_token_concat,temp_token_concat)
        cross_4 = self.cross_att4(temp_token_concat,imge,imge)
        cross_f3 = self.cross_attf3(cross_3,cross_4,cross_4)
        cross_f4 = self.cross_attf4(cross_4,cross_3,cross_3)
        corss_f3 = self.mlp3(cross_f3)
        corss_f4 = self.mlp3(cross_f4)

        sparse_embeddings =cross_f4[:,(-self.task_num):,:]
        # print(sparse_embeddings.shape)
        imge_fin = cross_3.permute(0,1,2).reshape(B,C,H,W)
        dense_embeddings = self.mask_adapter(imge_fin)
        # print(dense_embeddings.shape)

        return sparse_embeddings, dense_embeddings




class Samus(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.apg = AutoPromptGenerator(embed_dim = 256)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
          if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
            value.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
        fuse_apg: float = 1.0
    ) -> torch.Tensor:
        imge= self.image_encoder(imgs)

        se_apg,de_apg = self.apg(imge, self.mask_decoder.mask_tokens.weight)
        # print('A APG: ', se_apg.shape, de_apg.shape)
        if len(pt[0].shape) == 3:
          se_raw, de_raw = self.prompt_encoder(            # se b 2 256, de b 256 32 32
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
          # print(se.shape, se_apg.shape)
          se = fuse_apg*se_apg+(1-fuse_apg)*se_raw
          de = fuse_apg*de_apg+(1-fuse_apg)*de_raw
          low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                    image_embeddings=imge,
                    image_pe=self.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output= False,
                    )
          # print('B APG: ', se.shape, de.shape)
          masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
        else:
          low_res_masks, masks = [], []
          for i in range(pt[0].shape[1]):
            pti = (pt[0][:, i, :, :], pt[1][:, i, :])
            sei_raw, dei_raw = self.prompt_encoder(            # se b 2 256, de b 256 32 32
                        points=pti,
                        boxes=None,
                        masks=None,
                    )
            sei = fuse_apg*se_apg+(1-fuse_apg)*sei_raw
            dei = fuse_apg*de_apg+(1-fuse_apg)*dei_raw
            low_res_masksi, _ = self.mask_decoder( # low_res_mask b 1 128 128
                    image_embeddings=imge,
                    image_pe=self.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sei,
                    dense_prompt_embeddings=dei, 
                    multimask_output=False,
                    )
            masksi = F.interpolate(low_res_masksi, (256, 256), mode="bilinear", align_corners=False)
            low_res_masks.append(low_res_masksi)
            masks.append(masksi)
          low_res_masks = torch.stack(low_res_masks, dim=1)
          masks = torch.stack(masks, dim=1) # b c 1 255 255
          masks = masks.reshape(masks.shape[0], -1, masks.shape[3], masks.shape[4])
          low_res_masks = low_res_masks.reshape(low_res_masks.shape[0], -1, low_res_masks.shape[3], low_res_masks.shape[4])
        outputs = {"low_res_logits": low_res_masks, "masks": masks}

        return outputs



    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class Same(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViTPP,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.seg_apg = AutoPromptGenerator(embed_dim = 256,task_num=2)
        self.hr_apg = AutoPromptGenerator(embed_dim = 256,task_num=4)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
          if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
            value.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
        fuse_apg: float = 1.0
    ) -> torch.Tensor:
        seg_imge,hr_imge= self.image_encoder(imgs)

        seg_se_apg,seg_de_apg = self.seg_apg(seg_imge, self.mask_decoder.mask_tokens.weight)
        hr_se_apg,hr_de_apg = self.hr_apg(hr_imge, self.mask_decoder.mask_tokens.weight)
        # print('A APG: ', se_apg.shape, de_apg.shape)
        seg_se_raw, seg_de_raw = self.prompt_encoder(            # se b 2 256, de b 256 32 32
                      points=None,
                      boxes=bbox,
                      masks=None,
                  )

        hr_se_raw, hr_de_raw = self.prompt_encoder(            # se b 2 256, de b 256 32 32
                      points=pt,
                      boxes=None,
                      masks=None,
                  )
        # print(seg_se_apg.shape,seg_se_raw.shape)
        # print(hr_se_apg.shape,hr_se_raw.shape)

        seg_se = fuse_apg*seg_se_apg+(1-fuse_apg)*seg_se_raw
        seg_de = fuse_apg*seg_de_apg+(1-fuse_apg)*seg_de_raw
        seg_low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                  image_embeddings=seg_imge,
                  image_pe=self.prompt_encoder.get_dense_pe(), 
                  sparse_prompt_embeddings=seg_se,
                  dense_prompt_embeddings=seg_de, 
                  multimask_output= False,
                  )
        # print('B APG: ', se.shape, de.shape)
        seg_masks = F.interpolate(seg_low_res_masks, (256, 256), mode="bilinear", align_corners=False)

        hr_se = fuse_apg*hr_se_apg+(1-fuse_apg)*hr_se_raw
        hr_de = fuse_apg*hr_de_apg+(1-fuse_apg)*hr_de_raw
        hr_low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                  image_embeddings=hr_imge,
                  image_pe=self.prompt_encoder.get_dense_pe(), 
                  sparse_prompt_embeddings=hr_se,
                  dense_prompt_embeddings=hr_de, 
                  multimask_output= True,
                  )
        # print('B APG: ', se.shape, de.shape)
        hr_masks = F.interpolate(hr_low_res_masks, (256, 256), mode="bilinear", align_corners=False)

        outputs = {"hr_low_res": hr_low_res_masks, "seg_masks": seg_masks, "seg_se": seg_se_raw, "hr_se": hr_se_raw, "seg_se_apg":seg_se_apg, "hr_se_apg":hr_se_apg}

        return outputs



    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
