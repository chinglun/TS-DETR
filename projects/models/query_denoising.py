# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox_xyxy_to_cxcywh
from .transformer import inverse_sigmoid
import numpy as np
import random


class DnQueryGenerator:

    def __init__(self,
                 num_queries,
                 hidden_dim,
                 num_classes,
                 noise_scale=dict(label=0.5, box=0.4),
                 group_cfg=dict(
                     dynamic=True, num_groups=None, num_dn_queries=None)):
        super(DnQueryGenerator, self).__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
    def _is_small_object(self, height, width):
        '''
        判断是否为小目标
        '''
        if height*width <= self.thresh:
            return True
        else:
            return False

    def _compute_overlap(self, bbox_a, bbox_b):
        '''
        计算重叠
        '''
        if bbox_a is None:
            return False
        left_max = torch.max(bbox_a[0], bbox_b[0])
        top_max = torch.max(bbox_a[1], bbox_b[1])
        right_min = torch.min(bbox_a[2], bbox_b[2])
        bottom_min = torch.min(bbox_a[3], bbox_b[3])
        inter = torch.max(torch.tensor(0.0, device='cuda:0'), (right_min-left_max)) * torch.max(torch.tensor(0.0, device='cuda:0'), (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def _donot_overlap(self, new_bbox, bboxes):
        '''
        是否有重叠
        '''
        for bbox in bboxes[0]:
            if self._compute_overlap(new_bbox, bbox):
                return False
        return True

    def _create_copy_annot(self, height, width, bbox, bboxes):
        '''
        创建新的标签
        '''
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        for epoch in range(100):
            random_x, random_y = np.random.randint(int(bbox_w / 2), int(width - bbox_w / 2)), \
                np.random.randint(int(bbox_h / 2), int(height - bbox_h / 2))
            tl_x, tl_y = random_x - bbox_w/2, random_y-bbox_h/2
            br_x, br_y = tl_x + bbox_w, tl_y + bbox_h
            if tl_x < 0 or br_x > width or tl_y < 0 or tl_y > height:
                continue
            #new_bbox = np.array([tl_x, tl_y, br_x, br_y]).astype(np.int32)
            new_bbox = np.array([tl_x.cpu().numpy(), tl_y.cpu().numpy(), br_x.cpu().numpy(), br_y.cpu().numpy()])
            new_bbox=torch.tensor(new_bbox,device="cuda:0")

            if not self._donot_overlap(new_bbox, bboxes):
                continue
            return new_bbox
        return None
    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def __call__(self,
                 gt_bboxes,
                 gt_labels=None,
                 label_enc=None,
                 img_metas=None):
        """

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,
                TODO:noisy_label would be None.

        Returns:
            TODO
        """
        # TODO: temp only support for CDN
        # TODO: temp assert gt_labels is not None and label_enc is not None
        # print(gt_bboxes,gt_labels)
        # input("")
        # print(img_metas)
        # input("")
        #new_bboxes_list = []
        # gt_bboxes_copy=gt_bboxes
        # gt_labels_copy=gt_labels
        # for k,(img_meta, bboxes,labels) in enumerate(zip(img_metas, gt_bboxes, gt_labels)):
        #     for i in range(5):
        #         random_bbox = random.choice(bboxes)
        #         #print(random_bbox)
        #         x1, y1, x2, y2 = random_bbox
        #         #print( x1, y1, x2, y2)
        #         # width = x2 - x1
        #         # height = y2 - y1
        #         # print(x1, y1, x2, y2)
        #         # input("")
        #         img_height = torch.tensor(img_meta['img_shape'][0])
        #         img_width = torch.tensor(img_meta['img_shape'][1])
        #         #print(img_height, img_width, [x1, y1, x2, y2], gt_bboxes)
        #         new_bbox = self._create_copy_annot(img_height, img_width, [x1, y1, x2, y2], gt_bboxes_copy)
        #         new_bbox  = new_bbox.unsqueeze(0)
        #         print(new_bbox)
        #         input("")
        # #     print(new_bbox)
        #         if torch.numel(new_bbox) != 0:
        #             random_number = torch.tensor(random.randint(self.num_classes-1, self.num_classes-1),device="cuda:0")
        #             random_number = random_number.unsqueeze(0)
        #             #print(random_number)
        #             #input("")
        #             gt_bboxes_copy[k] = torch.cat([gt_bboxes_copy[k], new_bbox], dim=0)
        #             gt_labels_copy[k] = torch.cat([gt_labels_copy[k], random_number], dim=0)


        # gt_bboxes=gt_bboxes_copy
        # gt_labels=gt_labels_copy
        # print(gt_bboxes,gt_labels)
        # input("")

        # random_number = torch.tensor(random.randint(0, self.num_classes),device="cuda:0")
        # print(gt_labels)
        # gt_labels.append(random_number)
        # print(gt_labels)
        # input("")
        


        # #test random gt_labels
        # for tensor in gt_labels:
        #     # 生成随机整数，范围是从1到90
        #     random_values = torch.randint(low=self.num_classes-1, high=self.num_classes+1, size=tensor.size(), device=tensor.device)
        #     # 将生成的随机整数替换到张量中
        #     tensor.copy_(random_values)
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'
        assert gt_labels is not None \
               and label_enc is not None \
               and img_metas is not None  # TODO: adjust args
        batch_size = len(gt_bboxes)

        # convert bbox
        gt_bboxes_list = []
        for img_meta, bboxes in zip(img_metas, gt_bboxes):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes) / factor
            gt_bboxes_list.append(bboxes_normalized)
        gt_bboxes = gt_bboxes_list

        known = [torch.ones_like(labels) for labels in gt_labels]
        # print(known)
        # input("")
        known_num = [sum(k) for k in known]
        # print(known_num)
        # input("")
        num_groups = self.get_num_groups(int(max(known_num)))
        # print(num_groups)
        # input("")
        unmask_bbox = unmask_label = torch.cat(known)
        # print(unmask_bbox)
        # input("")
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes)
        # print(boxes)
        # input("")
        batch_idx = torch.cat(
            [torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])
        # print(batch_idx)
        # input("")
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        # print(known_indice)
        # input("")
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * num_groups, 1).view(-1)
        # print(known_indice)
        # input("")
        known_labels = labels.repeat(2 * num_groups, 1).view(-1)
        # print(known_labels)
        # input("")
        known_bid = batch_idx.repeat(2 * num_groups, 1).view(-1)
        # print(known_bid)
        # input("")
        known_bboxs = boxes.repeat(2 * num_groups, 1)
        # print(known_bid)
        # input("")
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expand.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))  # TODO

        pad_size = int(single_pad * 2 * num_groups)
        positive_idx = torch.tensor(range(
            len(boxes))).long().cuda().unsqueeze(0).repeat(num_groups, 1)
        positive_idx += (torch.tensor(range(num_groups)) * len(boxes) *
                         2).long().cuda().unsqueeze(1)
        # print(positive_idx)
        # input("")
        positive_idx = positive_idx.flatten()
        # print(positive_idx)
        # input("")
        negative_idx = positive_idx + len(boxes)
        # print(negative_idx)
        # input("")
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32)
            rand_sign = rand_sign * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ += \
                torch.mul(rand_part, diff).cuda() * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expand.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand, eps=1e-3)

        padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(2 * num_groups)
            ]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(),
                               map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(),
                              map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
            if i == num_groups - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': num_groups,
        }
        return input_query_label, input_query_bbox, attn_mask, dn_meta


class CdnQueryGenerator(DnQueryGenerator):

    def __init__(self, *args, **kwargs):
        super(CdnQueryGenerator, self).__init__(*args, **kwargs)


def build_dn_generator(dn_args):
    """

    Args:
        dn_args (dict):

    Returns:

    """
    if dn_args is None:
        return None
    type = dn_args.pop('type')
    if type == 'DnQueryGenerator':
        return DnQueryGenerator(**dn_args)
    elif type == 'CdnQueryGenerator':
        return CdnQueryGenerator(**dn_args)
    else:
        raise NotImplementedError(f'{type} is not supported yet')