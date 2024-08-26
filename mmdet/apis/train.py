# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
from projects import *
from mmcv.runner import EvalHook as BaseEvalHook
import json
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import cv2
from mmcv.runner import Hook



def load_gt(annotations):
    s=0
    m=0
    l=0
    dict={}
    with open(annotations, 'r') as f:
        js= json.load(fp = f)
    label=0
    for dic in js['annotations']:
        label+=1
        dict[str(label)]=dic['bbox']
        dict[str(label)].append(dic['category_id']-1)
        dict[str(label)].append(dic['image_id'])
        dict[str(label)].append(0)
    for dic in js['images']:
        for x,y in dict.items():
            #print(y[5])
            if str(y[5])==str(dic['id']):
               dict[str(x)].append(dic['file_name']) 
    for value in dict.values():
        if value[2]*value[3]>=96*96:
            l+=1
        elif value[2]*value[3]>=32*32 and value[2]*value[3]<96*96:
            m+=1
        else :
            s+=1
    #print(dict)#x1 y1 w h class id flag jpg
    return s,m,l,dict

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)

    iou_area = iou_w * iou_h
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    gt_area=bbox_gt[2]*bbox_gt[3]
    return max(iou_area/all_area, 0)

# def crop_area(ann,bboxes,filename,img):
#     x1=0
#     if bboxes is not None:
#         for  i,bbox in enumerate(bboxes):
#             flag=0
#             boxA=[int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])]
#             for value in ann.values():
#                 if str(value[7])==filename:
#                     boxB=[int(value[0]),int(value[1]),int(value[2]),int(value[3])]
#                     iou=get_iou(boxA,boxB)
#                     #print(iou)
#                     if iou>=0.3:
#                         x1+=1
#                         #cv2.imwrite('crop_eval_circle_aug'+"/"+str(value[4])+"/"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
#                         flag=1
#                         break
#             if flag==0:
#                 x1+=1
#                 cv2.imwrite('fp'+"/"+'Negative Sample'+"/"+filename+"_"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
#                 x1+=1
#                 #print(filename,x1)

def crop_negative_confused(ann,bboxes,filename,img,pred_class,score,confused):
    x1=0
    if bboxes is not None:
        for  i,bbox in enumerate(bboxes):
            flag=0
            boxA=[int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])]
            for value in ann.values():
                if str(value[7])==filename:
                    boxB=[int(value[0]),int(value[1]),int(value[2]),int(value[3])]
                    iou=get_iou(boxA,boxB)
                    if iou>=0.3 and int(pred_class[i])!=int(value[4]):
                        #confused[int(pred_class[i])]=confused[int(pred_class[i])]+1
                        confused[int(value[4])]=confused[int(value[4])]+1

                    #print(iou)
                    if iou>=0.3:
                        x1+=1
                        #cv2.imwrite('crop_eval_circle_aug'+"/"+str(value[4])+"/"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
                        flag=1
                        break
            if flag==0:
                x1+=1
                cv2.imwrite('fp'+"/"+'Negative Sample'+"/"+filename+"_"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
                x1+=1
                #print(filename,x1)
    with open("fp/confused.txt", "w") as file:
        for index, value in enumerate(confused):
            file.write(f"{index}:{value}\n")

class MyNegativeSampleHook(Hook):
    def after_epoch(self, runner):
        current_epoch = runner.epoch
        # if current_epoch % 1 == 0: 
        if current_epoch % 5 == 0 and current_epoch!= 0 :
             # 清空Negative Sample目录中的所有照片
            folder = 'fp/Negative Sample'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            model1 = init_detector('projects/configs/co_dino/MutiAugment98.py', find_latest_checkpoint(runner.work_dir), device="cuda:0")
            score_thr = 0.3
            annotations = 'data/coco/annotations/instances_train2017.json'
            small_gt, medium_gt, large_gt, dict_ann = load_gt(annotations)
            train_dataset = "data/coco/train2017"
            # class_number=('1','2','3')
            class_number= ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
           '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
           '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
           '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
           '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
           '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
           '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
           '90', '91', '92', '93', '94', '95', '96','97')#98
            #class_number=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","97","98","99","100","101","102","103","104","105","106","107","108","109","110","111","112","114","115","116","118","119","120","121","122","123","124","125","126","127","128","129","130","131","132","133","134","135","136","137","139","140","141","142","144","145","146","147","148","150","151","155","156","157","158","159","160","163","164","165","167","168","169","170","172","175","176","177","181"]
           # class_number=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42"]
            confused=[]
            for i in range(98):#all folder num(182,43,3,98)
                confused.append(int(0))
            for filename in os.listdir(train_dataset):
                img = cv2.imread(train_dataset+"/"+filename)
                result = inference_detector(model1, train_dataset+"/"+filename)
                bboxes = np.vstack(result)
                pred_class = np.array([])
                for i in range(98):#class num(162,43,3,98)
                    for j in range(len(result[i])):
                        pred_class = np.append(pred_class, i)

                if score_thr > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    pred_class = pred_class[inds]
                    confidence=scores[inds]
                    pred_class1=[]
                    for i in range(len(confidence)):
                        pred_class1.append(class_number[pred_class[i].astype(int)])
                        # print(pred_class1)
                    # for i in range(len(confidence)):
                    #     print(pred_class1[i],confidence[i])
                    pred_class=pred_class1
                    # print(pred_class,filename)
                crop_negative_confused(dict_ann, bboxes, filename, img,pred_class,scores,confused)
                #crop_area(dict_ann, bboxes, filename, img)

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

   # negative sample selection
    my_negative_sample_hook = MyNegativeSampleHook()
    runner.register_hook(my_negative_sample_hook)


    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


