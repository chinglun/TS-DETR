# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from projects import *
import os,cv2
import json
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
        
def missing_error(ann,bboxes,filename,pred_class):
    corr=0
    acc_b_c=0
    s=0
    m=0
    l=0
    if bboxes is not None:
        for  i,bbox in enumerate(bboxes):
            #print(bbox)
            boxA=[int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])]
            for value in ann.values():
                if str(value[7])==filename:
                    boxB=[int(value[0]),int(value[1]),int(value[2]),int(value[3])]
                    iou=get_iou(boxA,boxB)
                    #print(iou)
                    #print(value[4],int(pred_class[i]))
                    if iou>=0.5 and int(value[4])==int(pred_class[i]):
                        acc_b_c+=1
                        value[6]=1
                    if iou>=0.5:
                        corr+=1
                        #value[6]=1
                        if value[2]*value[3]>=96*96:
                            l+=1
                        elif value[2]*value[3]>=32*32 and value[2]*value[3]<96*96:
                            m+=1
                        else :
                            s+=1
                        break
                    #print(filename)
    else:
        return 0
    return corr,s,m,l,ann,acc_b_c

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

def crop_area(ann,bboxes,filename,img,x):
    x1=x
    if bboxes is not None:
        for  i,bbox in enumerate(bboxes):
            flag=0
            boxA=[int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])]
            for value in ann.values():
                if str(value[7])==filename:
                    boxB=[int(value[0]),int(value[1]),int(value[2]),int(value[3])]
                    iou=get_iou(boxA,boxB)
                    #print(iou)
                    if iou>=0.35:
                        x1+=1
                        cv2.imwrite('crop_eval_circle_aug'+"/"+str(value[4])+"/"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
                        flag=1
                        break
            if flag==0:
                x1+=1
                cv2.imwrite('crop_eval_circle_aug'+"/"+'unknown'+"/"+str(x1)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
                print(filename,x1)
    else:
        return x1
    return x1

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('folder', help='folder file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a folder
    score_thr =0.35
    small_gt=0
    medium_gt=0
    large_gt=0
    small=0
    medium=0
    large=0
    accuracy=0
    #label=0
    annotations='data/coco/annotations/instances_val2017.json'
    #annotations='data/coco/annotations/snow.json'
    #annotations='data/coco/annotations/multi-calss_val.json'
    #annotations='eval_data_97/coco/annotations/instances_val2017.json'
    small_gt,medium_gt,large_gt,dict_ann=load_gt(annotations)
    pred=0
    labels=len(dict_ann)
    correct=0
    x=0
    #class_number=['0','1','2']
    class_number=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
           '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
           '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
           '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
           '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
           '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
           '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
           '90', '91', '92', '93', '94', '95', '96','97']#98
    # class_number=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    #        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
    #        '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
    #        '35', '36', '37', '38', '39', '40', '41', '42']#43
    #class_number=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","97","98","99","100","101","102","103","104","105","106","107","108","109","110","111","112","114","115","116","118","119","120","121","122","123","124","125","126","127","128","129","130","131","132","133","134","135","136","137","139","140","141","142","144","145","146","147","148","150","151","155","156","157","158","159","160","163","164","165","167","168","169","170","172","175","176","177","181"]
    #print(len(dict_ann))
    for filename in os.listdir(args.folder):
        #print(filename)
        img = cv2.imread(args.folder+"/"+filename)
        result = inference_detector(model, args.folder+"/"+filename)
        bboxes = np.vstack(result)
        pred_class= np.array([])
        for i in range(98):#162
            for j in range(len(result[i])):
                pred_class = np.append(pred_class,i)
            
        #print(bboxes)
        
        # show the results
        if score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            confidence=scores[inds]
            pred_class=pred_class[inds]
            pred_class1=[]
            for i in range(len(confidence)):
                pred_class1.append(class_number[pred_class[i].astype(int)])
                # print(pred_class1)
            # for i in range(len(confidence)):
            #     print(pred_class1[i],confidence[i])
            pred_class=pred_class1
            # print(pred_class,filename)

        ###
        # if bboxes is not None:
        #     num_bboxes = bboxes.shape[0]
        #     for  i,bbox in enumerate(bboxes):
        #         label+=1
        #         print(i,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
        #         cv2.imwrite('test'+"/"+str(label)+".png",img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
        #for crop area
        ###

        pred=pred+len(bboxes)
        corr,s,m,l,dict_ann,acc_b_c=missing_error(dict_ann,bboxes,filename,pred_class)
        correct=correct+corr
        small=small+s
        medium=medium+m
        large=large+l
        accuracy=accuracy+acc_b_c
        x1=crop_area(dict_ann,bboxes,filename,img,x)
        x=x1
        # show_result_pyplot(
        # model,
        # args.folder+"/"+filename,
        # result,
        # palette=args.palette,
        # score_thr=args.score_thr,
        # out_file=args.out_file+"/"+filename)

    ###
    for  value in dict_ann.values():
        if value[6]==0:
            print(value)
    ##show fp
    ###
    print('label:',labels)
    print('pred:',pred)
    print('correct:',correct)
    print('accuracy',accuracy)
    print("precision:",correct/pred)
    print("recall",correct/labels)
    precision=correct/pred
    recall=correct/labels
    print("f1:",2*precision*recall/(precision+recall))
    print('small/small_gt',small,small_gt)
    print('medium/medium_gt',medium,medium_gt)
    print('large/large_gt',large,large_gt)
    print('large/large_gt',large,large_gt)
    print(x)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
