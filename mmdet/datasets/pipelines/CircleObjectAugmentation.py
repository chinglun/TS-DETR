from ..builder import PIPELINES
import numpy as np
import random
from PIL import Image, ImageDraw
import cv2

@PIPELINES.register_module('CircleObjectAugmentation')
class CircleObjectAugmentation :

    def __init__(self, thresh=64*64, prob=0.5, copy_times=10, epochs=100, all_objects=False, one_object=False):
        '''
        SmallObjectAugmentation: https://arxiv.org/abs/1902.07296
        https://github.com/zzl-pointcloud/Data_Augmentation_Zoo_for_Object_Detection/blob/master/augmentation_zoo/SmallObjectAugmentation.py
        args:
            thresh: small object thresh
            prob: the probability of whether to augmentation
            copy_times: how many times to copy anno
            epochs: how many times try to create anno
            all_object: copy all object once
            one_object: copy one object
        '''
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

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
        left_max = max(bbox_a[0], bbox_b[0])
        top_max = max(bbox_a[1], bbox_b[1])
        right_min = min(bbox_a[2], bbox_b[2])
        bottom_min = min(bbox_a[3], bbox_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def _donot_overlap(self, new_bbox, bboxes):
        '''
        是否有重叠
        '''
        for bbox in bboxes:
            if self._compute_overlap(new_bbox, bbox):
                return False
        return True

    def _create_copy_annot(self, height, width, bbox, bboxes):
        '''
        创建新的标签
        '''
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(bbox_w / 2), int(width - bbox_w / 2)), \
                np.random.randint(int(bbox_h / 2), int(height - bbox_h / 2))
            tl_x, tl_y = random_x - bbox_w/2, random_y-bbox_h/2
            br_x, br_y = tl_x + bbox_w, tl_y + bbox_h
            if tl_x < 0 or br_x > width or tl_y < 0 or tl_y > height:
                continue
            new_bbox = np.array([tl_x, tl_y, br_x, br_y]).astype(np.int32)

            if not self._donot_overlap(new_bbox, bboxes):
                continue

            return new_bbox
        return None
    # def crop_area(img,x,y,w,h):
    #     x2=[x+w/2,y]
    #     x1=[x,y+h/2]
    #     x3=[x+w,y+h/2]
    #     x4=[x+w/2,y+h]
    #     pts = np.array([x1,x2,x3,x4])
    #     pts=np.int32([pts])
    #     mask = np.zeros(img.shape[:2], np.uint8)
    #     cv2.polylines(mask, pts, 1, 255)    # 描绘边缘
    #     cv2.fillPoly(mask, pts, 255)
    #     dst = cv2.bitwise_and(img, img, mask=mask)
    #     return dst
    # def _add_patch_in_img(self, new_bbox, copy_bbox, image,height, width):
    #     '''
    #     复制图像区域
    #     '''
    #     flag=0
    #     copy_bbox = copy_bbox.astype(np.int32)
    #     # 打开背景图和长方形图
    #     rectangle_image = Image.fromarray(np.uint8(image))
    #     rectangle_image=rectangle_image.crop((copy_bbox[1],copy_bbox[0],copy_bbox[3],copy_bbox[2]))
    #     background_image = Image.fromarray(np.uint8(image))
    #     # 创建一个与背景图相同大小的画布
    #     composite_image = Image.new('RGB', (width,height))
    #     rectangle_width=new_bbox[2]-new_bbox[0]
    #     rectangle_height=new_bbox[3]-new_bbox[1]
    #     position_x = 0  # X坐标
    #     position_y = 0  # Y坐标
    #     # 将长方形图裁剪成椭圆
    #     ellipse_mask = Image.new('L', (rectangle_width, rectangle_height))
    #     draw = ImageDraw.Draw(ellipse_mask)
    #     draw.ellipse((0, 0, rectangle_width, rectangle_height), fill=255)
    #     # 裁剪长方形图
    #     rectangle_image = rectangle_image.crop((position_x, position_y, position_x + rectangle_width, position_y + rectangle_height))
    #     rectangle_image.putalpha(ellipse_mask)

    #     # 将长方形图粘贴到背景图的指定位置
    #     composite_image.paste(background_image , (0, 0))
    #     composite_image.paste(rectangle_image, (position_x, position_y), rectangle_image)
    #     flag=1
    #     return composite_image,flag

    def _add_patch_in_img(self, new_bbox, copy_bbox, image):
        '''
        复制图像区域
        '''
        flag=0
        copy_bbox = copy_bbox.astype(np.int32)
        try:
            # 打开背景图和长方形图
            rectangle_image = image
            rectangle_image=image[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
            #rectangle_image = cv2.GaussianBlur(rectangle_image, (35, 35), 0)
            #cv2.imwrite('composite_image.jpg', rectangle_image)
            background_image = image
            rectangle_width=copy_bbox[2]-copy_bbox[0]
            rectangle_height=copy_bbox[3]-copy_bbox[1]
            position_x = new_bbox[0]  # X坐标
            position_y = new_bbox[1]  # Y坐标

            # 创建一个与背景图相同大小的画布
            composite_image = background_image.copy()

            # # 创建一个椭圆掩码
            ellipse_mask = np.zeros((rectangle_height, rectangle_width), dtype=np.uint8)
            cv2.ellipse(ellipse_mask, (rectangle_width // 2, rectangle_height // 2), (rectangle_width // 2, rectangle_height // 2), 0, 0, 360, 255, -1)

            # # 将长方形图转换为灰度图
            # gray_rectangle = cv2.cvtColor(rectangle_image, cv2.COLOR_BGR2GRAY)

            # # Canny 边缘检测
            # edges = cv2.Canny(gray_rectangle, 30, 150)

            # # 查找轮廓
            # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # 创建一个与背景图像大小相同的掩膜
            # ellipse_mask = np.zeros_like(rectangle_image)

            # # 从最外部的轮廓开始逐渐缩小
            # for i in range(len(contours)):
            #     cv2.drawContours(ellipse_mask, contours, i, (255, 255, 255), thickness=i+1)

            #cv2.imwrite('result_image.jpg', ellipse_mask)
            # 合并长方形图到背景，使用椭圆掩码
            composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] = cv2.bitwise_and(rectangle_image, rectangle_image, mask=ellipse_mask)
            composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] += cv2.bitwise_and(background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], mask=~ellipse_mask)
            # if rectangle_height*rectangle_width>=96*96:
            #     kernel=10
            # elif rectangle_height*rectangle_width<32*32:
            #     kernel=3
            # else:
            #     kernel=5
                
            #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.medianBlur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], 3)
            #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.Blur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], (5,5))
            #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.medianBlur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], (5,5),0)
            #print("ooooooooooooooooooooo")
            flag=1
        except:
            return image,flag
        return composite_image,flag
    def __call__(self, results):
        if self.all_objects and self.one_object:
            return results
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']

        height, width = img.shape[0], img.shape[1]

        small_object_list = []
        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if self._is_small_object(bbox_h, bbox_w):
                small_object_list.append(idx)

        length = len(small_object_list)
        # 无小物体
        if 0 == length:
            return results

        # 随机选择不同的物体复制
        copy_object_num = np.random.randint(0, length)
        if self.all_objects:
            # 复制全部物体
            copy_object_num = length
        if self.one_object:
            # 只选择一个物体复制
            copy_object_num = 1

        random_list = random.sample(range(length), copy_object_num)
        idx_of_small_objects = [small_object_list[idx] for idx in random_list]
        select_bboxes = bboxes[idx_of_small_objects, :]
        select_labels = labels[idx_of_small_objects]

        bboxes = bboxes.tolist()
        labels = labels.tolist()
        for idx in range(copy_object_num):
            bbox = select_bboxes[idx]
            label = select_labels[idx]

            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if not self._is_small_object(bbox_h, bbox_w):
                continue

            for i in range(self.copy_times):
                new_bbox = self._create_copy_annot(height, width, bbox, bboxes)
                if new_bbox is not None:
                    img,flag = self._add_patch_in_img(new_bbox, bbox, img)
                    if flag==1:
                        bboxes.append(new_bbox)
                        labels.append(label)

        results['img'] = img
        results['gt_bboxes'] = np.array(bboxes).astype(np.float32)
        results['gt_labels'] = np.array(labels)
        return results