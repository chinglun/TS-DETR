import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import Transformer, DeformableDetrTransformer, DeformableDetrTransformerDecoder,DetrTransformerEncoder
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CoDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, look_forward_twice=False, **kwargs):

        super(CoDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class CoDeformableDetrTransformer(DeformableDetrTransformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 mixed_selection=True,
                 with_pos_coord=True,
                 with_coord_feat=True,
                 num_co_heads=1,
                 **kwargs):
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(CoDeformableDetrTransformer, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                # bug: this code should be 'self.head_pos_embed = nn.Embedding(self.num_co_heads, self.embed_dims)', we keep this bug for reproducing our results with ResNet-50.
                # You can fix this bug when reproducing results with swin transformer.
                self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1, self.embed_dims)
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims*2))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims*2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                return_encoder_output=False,
                attn_masks=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk = query_embed.shape[0]
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            if return_encoder_output:
                return inter_states, init_reference_out,\
                    inter_references_out, enc_outputs_class,\
                    enc_outputs_coord_unact, memory                
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        if return_encoder_output:
            return inter_states, init_reference_out, \
                inter_references_out, None, None, memory
        return inter_states, init_reference_out, \
            inter_references_out, None, None

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, \
            inter_references_out


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    # TODO: It can be implemented by add an out_channel arg of
    #  mmcv.cnn.bricks.transformer.FFN
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            pos_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000**(2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.embed_dims//2)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

@TRANSFORMER.register_module()
class CoDinoTransformer(CoDeformableDetrTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDinoTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)
    
    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.query_embed.weight.data)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        
        # print(topk_score)
        # input("")
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        # print(bs)
        # input("")
        # print(query)
        # print("")
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out, topk_score, topk_anchor, memory


    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
        memory = feat_flatten
            #enc_inter = [feat.permute(1, 2, 0) for feat in enc_inter]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = (pos_anchors)
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out
#modify
class RepVGGBlock(BaseModule):
    """RepVGG block is modifided to skip branch_norm.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): The stride of the block. Defaults to 1.
        padding (int): The padding of the block. Defaults to 1.
        dilation (int): The dilation of the block. Defaults to 1.
        groups (int): The groups of the block. Defaults to 1.
        padding_mode (str): The padding mode of the block. Defaults to 'zeros'.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to None.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict): The config dict for activation layers.
            Defaults to dict(type='ReLU').
        without_branch_norm (bool): Whether to skip branch_norm.
            Defaults to True.
        init_cfg (dict): The config dict for initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 conv_cfg = None,
                 norm_cfg= dict(type='BN'),
                 act_cfg= dict(type='ReLU'),
                 without_branch_norm: bool = True,
                 init_cfg= None):
        super(RepVGGBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # judge if input shape and output shape are the same.
        # If true, add a normalized identity shortcut.
        if out_channels == in_channels and stride == 1 and \
                padding == dilation and not without_branch_norm:
            self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.branch_norm = None

        self.branch_3x3 = self.create_conv_bn(
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )
        self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self,
                       kernel_size: int,
                       dilation: int = 1,
                       padding: int = 0) -> nn.Sequential:
        """Create a conv_bn layer.

        Args:
            kernel_size (int): The kernel size of the conv layer.
            dilation (int, optional): The dilation of the conv layer.
                Defaults to 1.
            padding (int, optional): The padding of the conv layer.
                Defaults to 0.

        Returns:
            nn.Sequential: The created conv_bn layer.
        """
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x) :
        """1x1 conv + 3x3 conv + identity shortcut.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out

        out = self.act(out)

        return out


class CSPRepLayer(BaseModule):
    """CSPRepLayer.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        num_blocks (int): The number of blocks in the layer. Defaults to 3.
        expansion (float): The expansion of the block. Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expansion: float = 1.0,
                 norm_cfg= dict(type='BN', requires_grad=True),
                 act_cfg = dict(type='SiLU', inplace=True)):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HybridEncoder(BaseModule):
    """HybridEncoder.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the layer.
        projector (:obj:`ConfigDict` or dict, optional): The config dict for
            the projector. Defaults to None.
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [512, 1024, 2048].
        feat_strides (List[int], optional): The strides of the feature
            maps. Defaults to [8, 16, 32].
        hidden_dim (int, optional): The hidden dimension of the MLP.
            Defaults to 256.
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        pe_temperature (int, optional): The temperature of the positional
            encoding. Defaults to 10000.
        expansion (float, optional): The expansion of the CSPRepLayer.
            Defaults to 1.0.
        depth_mult (float, optional): The depth multiplier of the CSPRepLayer.
            Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        eval_size (int, optional): The size of the test image.
            Defaults to None.
    """

    def __init__(self,
                 layer_cfg=None,
                 projector = None,
                 num_encoder_layers = 1,
                 in_channels = [512, 1024, 2048],
                 feat_strides= [8, 16, 32],
                 hidden_dim = 256,
                 use_encoder_idx= [2],
                 pe_temperature= 10000,
                 expansion = 1.0,
                 depth_mult= 1.0,
                 norm_cfg = dict(type='BN', requires_grad=True),
                 act_cfg = dict(type='SiLU', inplace=True),
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        # encoder transformer
        self.encoder = nn.ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        lateral_convs = list()
        fpn_blocks = list()
        for idx in range(len(in_channels) - 1, 0, -1):
            lateral_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.fpn_blocks = nn.ModuleList(fpn_blocks)

        # bottom-up pan
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(in_channels) - 1):
            downsample_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.downsample_convs = nn.ModuleList(downsample_convs)
        self.pan_blocks = nn.ModuleList(pan_blocks)

        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        proj_feats = [
            self.input_proj[i](inputs[i]) for i in range(len(inputs))
        ]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(
                    0, 2, 1).contiguous()
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](
                    src_flatten,
                    query_pos=pos_embed.to(src_flatten.device),
                    key_padding_mask=None)
                proj_feats[enc_ind] = memory.permute(
                    0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(
                feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.cat([downsample_feat, feat_high], axis=1))
            outs.append(out)

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)

    @staticmethod
    def build_2d_sincos_position_embedding(w: int,
                                           h: int,
                                           embed_dim=256,
                                           temperature=10000.) :
        """Build 2D sin-cos position embedding.

        Args:
            w (int): The width of the feature map.
            h (int): The height of the feature map.
            embed_dim (int): The dimension of the embedding.
                Defaults to 256.
            temperature (float): The temperature of the position embedding.
                Defaults to 10000.

        Returns:
            Tensor: The position embedding.
        """

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            ('Embed dimension must be divisible by 4 for '
             '2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                         axis=1)[None, :, :]
