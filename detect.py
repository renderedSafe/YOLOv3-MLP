from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    return img_


class Detector:
    def __init__(self, resolution=416):
        '''

        :param resolution: int, multiple of 32 greater than 32
        '''
        self.batch_size = 1
        self.scales = [1, 2, 3]
        self.resolution = resolution
        self.num_boxes = [self.resolution//8, self.resolution//16, self.resolution//32]
        self.num_boxes = sum([3*(x**2) for x in self.num_boxes])
        self.scales_indices = []
        for scale in self.scales:
            li = list(range((scale - 1)* self.num_boxes // 3, scale * self.num_boxes // 3))
            self.scales_indices.extend(li)
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.start = 0
        self.save_directory = '.'
        self.cfg_file = 'cfg/yolov3.cfg'
        self.weights_file = "yolov3.weights"
        self.colors = pkl.load(open("pallete", "rb"))

        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80
        self.classes = load_classes('data/coco.names')

        # Set up the neural network
        print("Loading network.....")
        self.model = Darknet(self.cfg_file)
        self.model.load_weights(self.weights_file)
        print("Network successfully loaded")

        self.model.net_info["height"] = self.resolution
        self.inp_dim = self.model.net_info["height"]
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()

        # Set the model in evaluation mode
        self.model.eval()

    def detect_objects(self, image_path):
        image_prep = prep_image(image_path, self.inp_dim)
        im_batches = [image_prep[0]]
        orig_ims = [image_prep[1]]
        im_dim_list = [image_prep[2]]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

        img_path = image_path

        if self.CUDA:
            im_dim_list = im_dim_list.cuda()

        write = False
        self.model(get_test_input(self.inp_dim, self.CUDA), self.CUDA)

        objs = {}
        i = 0
        for batch in im_batches:
            if self.CUDA:
                batch = batch.cuda()

            with torch.no_grad():
                prediction = self.model(Variable(batch), self.CUDA)

            prediction = prediction[:,self.scales_indices]

            prediction = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)
            prediction[:,0] += i*self.batch_size

            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output,prediction))

            for im_num, image in enumerate(img_path[i*self.batch_size: min((i +  1)*self.batch_size, len(img_path))]):
                im_id = i*self.batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")
            i += 1

            if self.CUDA:
                torch.cuda.synchronize()

        try:
            output
        except NameError:
            print("No detections were made")
            exit()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

        scaling_factor = torch.min(self.inp_dim/im_dim_list,1)[0].view(-1,1)

        output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

        def write(x, batches, results):
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            img = results[int(x[0])]
            cls = int(x[-1])
            label = "{0}".format(self.classes[cls])
            color = random.choice(self.colors)
            cv2.rectangle(img, c1, c2,color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2,color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            return img

        list(map(lambda x: write(x, im_batches, orig_ims), output))

        det_names = pd.Series(img_path).apply(lambda x: "{}/det_{}".format(self.save_directory,x.split("/")[-1]))

        cv2.imwrite(det_names[0], orig_ims[0])
        torch.cuda.empty_cache()
        ret_path = det_names[0]

        return ret_path, objs, orig_ims[0]
