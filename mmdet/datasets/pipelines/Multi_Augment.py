from ..builder import PIPELINES
import numpy as np
import random
from PIL import Image, ImageDraw
import cv2
import os
import glob

@PIPELINES.register_module('MutiAugment')
class MutiAugment :

    def __init__(self, thresh=3000*30000, prob=0.7, copy_times=1, epochs=100, copy_num=20,scale=True):
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
        self.copy_num=copy_num
        self.scale=scale


    def _is_small_object(self, height, width):
        '''
        '''
        if height*width <= self.thresh:
            return True
        else:
            return False

    def _compute_overlap(self, bbox_a, bbox_b):
        '''
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
        '''
        for bbox in bboxes:
            if self._compute_overlap(new_bbox, bbox):
                return False
        return True

    def _create_copy_annot(self, height, width, bbox, bboxes):
        '''
        '''
        try:
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
        except:
            #print(width,height,bbox_w,bbox_h,"skip")
            return None
    # def _add_patch_in_img(self, new_bbox, copy_bbox, image,sample):
    #     '''
    #     '''
    #     flag=0
    #     copy_bbox=np.array(copy_bbox)
    #     copy_bbox = copy_bbox.astype(np.int32)
    #     try:
    #         image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2],
    #             :] = sample[:,:,:]
    #         flag=1
    #     except:
    #         return image,flag
    #     return image,flag

    # def crop_area(img,x,y,w,h):
    #     x2=[x+w/2,y]
    #     x1=[x,y+h/2]
    #     x3=[x+w,y+h/2]
    #     x4=[x+w/2,y+h]
    #     pts = np.array([x1,x2,x3,x4])
    #     pts=np.int32([pts])
    #     mask = np.zeros(img.shape[:2], np.uint8)
    #     cv2.polylines(mask, pts, 1, 255)    
    #     cv2.fillPoly(mask, pts, 255)
    #     dst = cv2.bitwise_and(img, img, mask=mask)
    #     return dst
    # def _add_patch_in_img(self, new_bbox, copy_bbox, image,height, width):
    #     '''
    #     '''
    #     flag=0
    #     copy_bbox = copy_bbox.astype(np.int32)
    #     rectangle_image = Image.fromarray(np.uint8(image))
    #     rectangle_image=rectangle_image.crop((copy_bbox[1],copy_bbox[0],copy_bbox[3],copy_bbox[2]))
    #     background_image = Image.fromarray(np.uint8(image))
    #     composite_image = Image.new('RGB', (width,height))
    #     rectangle_width=new_bbox[2]-new_bbox[0]
    #     rectangle_height=new_bbox[3]-new_bbox[1]
    #     position_x = 0  
    #     position_y = 0   
    #     ellipse_mask = Image.new('L', (rectangle_width, rectangle_height))
    #     draw = ImageDraw.Draw(ellipse_mask)
    #     draw.ellipse((0, 0, rectangle_width, rectangle_height), fill=255)
    #     rectangle_image = rectangle_image.crop((position_x, position_y, position_x + rectangle_width, position_y + rectangle_height))
    #     rectangle_image.putalpha(ellipse_mask)

    #     composite_image.paste(background_image , (0, 0))
    #     composite_image.paste(rectangle_image, (position_x, position_y), rectangle_image)
    #     flag=1
    #     return composite_image,flag
    

    # def _add_patch_in_img(self, new_bbox, copy_bbox, image,sample):
    #     '''
    #     '''
    #     flag=0
    #     copy_bbox=np.array(copy_bbox)
    #     copy_bbox = copy_bbox.astype(np.int32)
    #     random_number = random.choice([1,1,2,2,3])
    #     try:
    #         rectangle_image =sample
    #         rectangle_image=sample[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
            
    #         #rectangle_image = cv2.GaussianBlur(rectangle_image, (35, 35), 0)
    #         #cv2.imwrite('composite_image.jpg', rectangle_image)
    #         background_image = image
    #         rectangle_width=copy_bbox[2]-copy_bbox[0]
    #         rectangle_height=copy_bbox[3]-copy_bbox[1]
    #         position_x = new_bbox[0]  
    #         position_y = new_bbox[1] 
            
    #         if random_number==2:
    #             alpha = np.random.uniform(0.5, 2.5) 
    #             beta = np.random.uniform(-75, 75)   
    #             rectangle_image = cv2.addWeighted(sample, alpha, np.zeros(sample.shape, dtype=sample.dtype), 0, beta)
    #         elif random_number==3:
    #              # noise
    #             salt_prob=0.05
    #             pepper_prob=0.1
    #             num_salt = np.ceil(salt_prob * rectangle_image.size)
    #             num_pepper = np.ceil(pepper_prob * rectangle_image.size)
    #             coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in rectangle_image.shape]
    #             coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in rectangle_image.shape]

    #             rectangle_image[coords_salt[0], coords_salt[1], :] = 1
    #             rectangle_image[coords_pepper[0], coords_pepper[1], :] = 0
                

    #         composite_image = background_image.copy()

    #         ellipse_mask = np.zeros((rectangle_height, rectangle_width), dtype=np.uint8)
    #         cv2.ellipse(ellipse_mask, (rectangle_width // 2, rectangle_height // 2), (rectangle_width // 2, rectangle_height // 2), 0, 0, 360, 255, -1)

    #         # gray_rectangle = cv2.cvtColor(rectangle_image, cv2.COLOR_BGR2GRAY)

    #         # edges = cv2.Canny(gray_rectangle, 30, 150)

    #         # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #         # ellipse_mask = np.zeros_like(rectangle_image)

    #         # for i in range(len(contours)):
    #         #     cv2.drawContours(ellipse_mask, contours, i, (255, 255, 255), thickness=i+1)

    #         #cv2.imwrite('result_image.jpg', ellipse_mask)
    #         composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] = cv2.bitwise_and(rectangle_image, rectangle_image, mask=ellipse_mask)
    #         composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] += cv2.bitwise_and(background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], mask=~ellipse_mask)
    #         # if rectangle_height*rectangle_width>=96*96:
    #         #     kernel=10
    #         # elif rectangle_height*rectangle_width<32*32:
    #         #     kernel=3
    #         # else:
    #         #     kernel=5
                
    #         #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.medianBlur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], 3)
    #         #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.Blur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], (5,5))
    #         #composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]=cv2.medianBlur(composite_image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], (5,5),0)
    #         #print("ooooooooooooooooooooo")
    #         flag=1
    #     except:
    #         return image,flag
    #     return composite_image,flag


    def _add_patch_in_img(self, new_bbox, copy_bbox, image,sample):
        '''

        '''
        flag=0
        copy_bbox=np.array(copy_bbox)
        copy_bbox = copy_bbox.astype(np.int32)
        try:
            rectangle_image =sample
            rectangle_image=sample[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
            background_image = image
            rectangle_width=copy_bbox[2]-copy_bbox[0]
            rectangle_height=copy_bbox[3]-copy_bbox[1]
            position_x = new_bbox[0]  
            position_y = new_bbox[1] 

            # if np.random.rand() > 0.5:
            #     rectangle_image=cv2.flip(rectangle_image,1)#add 20240105

            composite_image = background_image.copy()

            ellipse_mask = np.zeros((rectangle_height, rectangle_width), dtype=np.uint8)
            cv2.ellipse(ellipse_mask, (rectangle_width // 2, rectangle_height // 2), (rectangle_width // 2, rectangle_height // 2), 0, 0, 360, 255, -1)

            composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] = cv2.bitwise_and(rectangle_image, rectangle_image, mask=ellipse_mask)
            composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width] += cv2.bitwise_and(background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], background_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width], mask=~ellipse_mask)
            flag=1
        except:
            return image,flag
        return composite_image,flag
    # def _add_patch_in_img(self, new_bbox, copy_bbox, image,sample):
    #     '''

    #     '''
    #     flag=0
    #     copy_bbox=np.array(copy_bbox)
    #     copy_bbox = copy_bbox.astype(np.int32)
    #     try:
    #         rectangle_image =sample
    #         rectangle_image=sample[copy_bbox[1]:copy_bbox[3], copy_bbox[0]:copy_bbox[2], :]
    #         background_image = image
    #         rectangle_width=copy_bbox[2]-copy_bbox[0]
    #         rectangle_height=copy_bbox[3]-copy_bbox[1]
    #         position_x = new_bbox[0]
    #         position_y = new_bbox[1]

    #         composite_image = background_image.copy()
    #         mask = np.all(rectangle_image != [0, 0, 0], axis=-1)
    #         composite_image[position_y:position_y + rectangle_height, position_x:position_x + rectangle_width][mask] = rectangle_image[mask]
    #         flag=1
    #     except:
    #         return image,flag
    #     return composite_image,flag
    
    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        #class_number=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","97","98","99","100","101","102","103","104","105","106","107","108","109","110","111","112","114","115","116","118","119","120","121","122","123","124","125","126","127","128","129","130","131","132","133","134","135","136","137","139","140","141","142","144","145","146","147","148","150","151","155","156","157","158","159","160","163","164","165","167","168","169","170","172","175","176","177","181"]
        class_number=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
           '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
           '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
           '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
           '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
           '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
           '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
           '90', '91', '92', '93', '94', '95', '96','97']#98
        #class_number=['0','1','2']
        # class_number=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
        #    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
        #    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
        #    '35', '36', '37', '38', '39', '40', '41', '42']#43
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


        categories = []
        quantities = []  
        root="crop-twtsdb"
        #root="crop-tt100k-segment"
        #root="crop-tt100k"
        #root="crop-gtsdb"
        for filename in os.listdir(root):
            num=0
            for _ in os.listdir(os.path.join(root, filename)):
                num+=1
            if num!=0:
                categories.append(filename)
                quantities.append(num)


        confused=[]
        if os.path.exists("fp/confused.txt"):
            with open("fp/confused.txt", "r") as file:
                for line in file:
                    index, value = line.strip().split(":")
                    confused.append(int(value))
            # for i, value in enumerate(confused):
            #     if int(value)!=0:
            #         index = categories.index(str(i))
            #         confused[i]=confused[i]/quantities[index]



            # 找出前10个最大值的索引
            top_10_indices = np.argsort(confused)[-10:][::-1]
            # print(top_10_indices)
            # input("")


        



        bias=100
        probabilities = [1 / (quantity+bias) for quantity in quantities]
        if os.path.exists("fp/confused.txt"):
            max_prob=max(probabilities)
            for i in top_10_indices:
                index = categories.index(str(i))
                probabilities[int(index)]=max_prob

        #print(probabilities)
        #print(probabilities)
        selected_category=[]
        for _ in  range(self.copy_num):
            selected_category.append(random.choices(categories, probabilities)[0])

        select_img=[]
        select_bboxes=[] 
        select_labels=[]
        for category in selected_category:
            select_img.append(random.choice(os.listdir(os.path.join(root, category))))
            copy_sample=cv2.imread(os.path.join(root,category,select_img[-1]))
            h, w = copy_sample.shape[0], copy_sample.shape[1]
            if self.scale:
                scale_factor_h = random.uniform(0.7, 1.414)
                scale_factor_w = random.uniform(0.7, 1.414)
                h, w = int(h * scale_factor_h), int(w * scale_factor_w)
            select_labels.append(int(class_number.index(category)))
            select_bboxes.append([0,0,h,w])

        # with open("show.txt", "w") as file:
        #     for item in select_img:
        #         file.write(f"{1}\n")

        bboxes = bboxes.tolist()
        labels = labels.tolist()
        
        for idx in range(self.copy_num):
            bbox = select_bboxes[idx]
            label = select_labels[idx]
            sample=select_img[idx]
            sample=cv2.imread(os.path.join(root,str(class_number[label]),sample))
            bbox_h, bbox_w = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if self.scale:
                sample=cv2.resize(sample,(int(bbox_h),int(bbox_w)))


            if not self._is_small_object(bbox_h, bbox_w):
                continue

            for i in range(self.copy_times):
                new_bbox = self._create_copy_annot(height, width, bbox, bboxes)
                if new_bbox is not None:
                    img,flag = self._add_patch_in_img(new_bbox, bbox, img,sample)
                    if flag==1:
                        bboxes.append(new_bbox)
                        # if np.random.rand() > 0.2:
                        labels.append(label)
                        # else:
                        #labels.append(int(random.choices(categories)[0]))
                        


        # #copy for false positive
        select_fp=[]
        unknown_fp = os.path.join("fp", "Negative Sample")
        if os.path.exists(unknown_fp) and os.listdir(unknown_fp):
            for i in range(5):
                select_fp.append(random.choice(os.listdir(os.path.join("fp", "Negative Sample"))))
                copy_fp=cv2.imread(os.path.join("fp","Negative Sample",select_fp[-1]))
                h, w = copy_fp.shape[0], copy_fp.shape[1]
                if self.scale:
                    scale_factor_h = random.uniform(0.7, 10)
                    scale_factor_w = random.uniform(0.7, 10)
                    h, w = int(h * scale_factor_h), int(w * scale_factor_w)
                    copy_fp=cv2.resize(copy_fp,(int(h),int(w)))
                for i in range(self.copy_times):
                    new_bbox = self._create_copy_annot(height, width, [0,0,h,w], bboxes)
                    if new_bbox is not None:
                        img,flag = self._add_patch_in_img(new_bbox, bbox, img,copy_fp)

                    

        results['img'] = img
        results['gt_bboxes'] = np.array(bboxes).astype(np.float32)
        results['gt_labels'] = np.array(labels)
        return results