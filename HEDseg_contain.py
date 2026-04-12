from HEDseg_ui import Ui_HEDseg
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import os 
import cv2 
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_HEDseg()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.FolderSelection_src.clicked.connect(self.get_src)
        self.ui.FolderSelection_dst.clicked.connect(self.get_dst)
        self.ui.ModelcomboBox.currentIndexChanged.connect(self.get_network)
        self.ui.BackbonecomboBox.currentIndexChanged.connect(self.get_backbone)
        self.ui.WeightSelection.clicked.connect(self.get_weight)
        self.ui.Previous.clicked.connect(self.get_previous)
        self.ui.Next.clicked.connect(self.get_next)
        self.ui.Display.clicked.connect(self.get_predict)
        self.ui.Reset.clicked.connect(self.reset_predict)
        self.ui.checkBox_1.clicked.connect(self.set_contour_1)
        self.ui.checkBox_2.clicked.connect(self.set_contour_2)
        self.ui.horizontalSlider.valueChanged.connect(self.get_threshold)
        
        #
        self.src_list = []
        self.img_idx = 0 
        self.dst_path = r''
        self.network = r'---'
        self.backbone = r'---'
        self.model_weight = r''
        self.model_status = 0 # 0:not ready / 1:ready
        self.img_in_status = 0 # 0:not ready / 1:ready
        self.contour_state = [0, 0]
        self.threshold = 0.0
        
    def get_src(self):
        
        self.folder_path = QFileDialog.getExistingDirectory(self,
                    "Open folder",
                    "./")

        if self.folder_path != None:
            img_names=os.listdir(self.folder_path)
            for img_name in (img_names):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    self.src_list.append(self.folder_path+'/'+ img_name)
            print('{} is selected as source path ({} images are found)'.format(self.folder_path, len(self.src_list)))
            print('Input Image Status : Ready')
            
            ###load image of index 0
            self.img_in = cv2.imread(self.src_list[self.img_idx],1)
            cv2.imshow('Input Image', self.img_in)
            self.img_in_status = 1
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
    def get_dst(self):
        
        self.folder_path = QFileDialog.getExistingDirectory(self,
                    "Open folder",
                    "./")
        self.dst_path = self.folder_path
        print('{} is selected as output path'.format(self.dst_path))
        
    def get_network(self):
        
        self.network = self.ui.ModelcomboBox.currentText()
        
    def get_backbone(self):
        
        self.backbone = self.ui.BackbonecomboBox.currentText()
        
    def get_weight(self):
        
        filename, filetype = QFileDialog.getOpenFileName(self,
                    "Open folder",
                    "./")
        if filename.endswith('.pth'):
            self.model_weight = filename
            print('{} is chosen as model wieght...'.format(filename.split('/')[-1]))
            if self.network != r'---' and self.backbone != r'---':
                self.model = smp.Unet(encoder_name=self.backbone,
                                           decoder_channels=(512, 256, 64, 32, 16),
                                           in_channels=3,
                                           classes=3)
                (self.model).cuda()
                self.model.load_state_dict(torch.load(self.model_weight))
                self.model_status = 1
                print('Model Status : Ready')
                
            else:
                self.model_status = 0
                print('Model Status : Not ready')
        else:
            print('File format is incorrect, please select again')
            
    def get_previous(self):
        
        if self.img_idx > 0:
            self.img_idx = self.img_idx - 1
            self.img_in = cv2.imread(self.src_list[self.img_idx],1)
            cv2.imshow('Input Image', self.img_in)
            
        else:
            pass
        
    def get_next(self):
        
        if self.img_idx < (len(self.src_list) - 1):
            self.img_idx = self.img_idx + 1
            self.img_in = cv2.imread(self.src_list[self.img_idx],1)
            cv2.imshow('Input Image', self.img_in)
            
        else:
            pass
    
    '''Non-trigger Function Section'''
    
    def toTensor(self,src):
        img_rgb = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
        output = torch.tensor(img_rgb / 255)
        output = output.float()
        output = torch.permute(output,(2,0,1))
        output = torch.unsqueeze(output, 0)
        return output
    
    def toMask(self,src):
        output = torch.argmax(src, dim=1)
        output = torch.squeeze(output, 0)
        output = output.to(torch.uint8)
        return output
    
    def get_class(self,src,class_type=0):
        output = np.where(src == class_type , 255, 0)
        output = np.uint8(output)
        return output

    def get_bg(self,src,iterations=2):
        kernel = np.ones((3,3),np.uint8)
        output = cv2.dilate(src,kernel,iterations=iterations)
        return output

    def dist_trans(self,src):
        kernel = np.ones((2,2),np.uint8)
        dist_transform = cv2.distanceTransform(src,cv2.DIST_L2,maskSize=0)
        dist_output = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        dist_output = cv2.filter2D(dist_output,-1,kernel)
        return dist_output
        
    def get_seed(self,src,factor=1.0): # binary image as input
        _, region = cv2.connectedComponents(src) # 0 = background
        dist_img = self.dist_trans(src)
        region_num = len(np.unique(region))
        output = np.zeros((512,512),dtype=int)
        for i in range(1,region_num):
            sub_mask = np.where(region == i, 1, 0)
            sub_dist_img = dist_img * sub_mask
            _, sub_fg = cv2.threshold(sub_dist_img, factor * sub_dist_img.max(),255,0)
            output = output + sub_fg
        output = np.uint8(output)
        return output

    def get_seg_mask(self,src):
        output = np.where(src <= 1 , 0, 255)
        output = np.uint8(output)
        return output

    def seg_class(self,src,class_type=0,factor=1.0):
        # Preparation
        class_mask = self.get_class(src,class_type=class_type)
        class_fg, class_bg = self.get_seed(class_mask,factor=factor), self.get_bg(class_mask,iterations=1)
        class_unknown = cv2.subtract(class_bg,class_fg)
        # Watersheld
        _, class_markers = cv2.connectedComponents(class_fg)
        class_markers = class_markers + 1
        class_markers = np.where(class_unknown==255,0,class_markers)
        class_mask = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2BGR)
        class_markers = cv2.watershed(class_mask,class_markers)
        class_seg = self.get_seg_mask(class_markers)
        # Find contours
        class_contours, _ = cv2.findContours(class_seg,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return class_seg, class_contours

    def join_contours(self,src,contour1,contour2,width=1, state=[0, 0]):
        
        if state == [0, 0]:
            output = src
        
        elif state == [1, 0]:
            output = cv2.drawContours(src, contour1, -1, (0,0,255), width)
            
        elif state == [0, 1]:
            output = cv2.drawContours(src, contour2, -1, (255,0,0), width)
            
        else:
            output = cv2.drawContours(src, contour1, -1, (0,0,255), width)
            output = cv2.drawContours(output, contour2, -1, (255,0,0), width)
            
        return output
    
    '''============================'''
    
    def get_predict(self):
        
        if self.model_status == 1 and self.img_in_status == 1:
            
            # mask prediction
            img_norm = self.toTensor(self.img_in)
            img_norm = img_norm.cuda()
            output = self.model(img_norm)
            mask = self.toMask(output)
            mask = mask.cpu()
            mask = mask.numpy()
            
            # watershed
            _, contours_red = self.seg_class(mask,class_type=1,factor=self.threshold) # slider contorl
            _, contours_blue = self.seg_class(mask,class_type=2,factor=self.threshold) # slider contorl
            result = self.join_contours((self.img_in).copy(), contours_red, contours_blue,width=2,state=self.contour_state)
            cv2.imshow('Segmentation Image', result)
        
        else:
            print('Model or input image is not ready...')
            
            
    def reset_predict(self):
        
        if self.img_in_status == 1:
            reset_img = self.img_in
            self.ui.checkBox_1.setChecked(False)
            self.ui.checkBox_2.setChecked(False)
            self.contour_state = [0, 0]
            cv2.imshow('Segmentation Image', reset_img)
            
        else:
            pass
        
        
    def set_contour_1(self):
        
        if self.ui.checkBox_1.isChecked():
            self.contour_state[0] = 1
        
        else:
            self.contour_state[0] = 0
            
    def set_contour_2(self):
        
        if self.ui.checkBox_2.isChecked():
            self.contour_state[1] = 1
        
        else:
            self.contour_state[1] = 0
            
    def get_threshold(self):
        
        self.threshold = round(self.ui.horizontalSlider.value() * 0.1, 2)
        self.get_predict()