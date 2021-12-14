import os
import numpy as np
import matplotlib.pyplot as plt
from kitti_utils.calibration import Calibration
class kitti_data(object):
    '''Load and parse object data into a usable format.'''
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        #
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.image_dir = os.path.join(self.split_dir, 'image_2')

        self.calib=Calibration(root_dir, split)

    def get_image(self, idx):
        image_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return self.load_camera_image_2(image_filename)
    def load_camera_image_2(self,image_filename):
        return plt.imread(image_filename)

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return self.load_velo_scan(lidar_filename)

    def load_velo_scan(self,velo_filename):
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_calibration(self, idx):
        self.calib.read_cal_file(idx)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return self.read_label(label_filename)

    def read_label(self,label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3d(line) for line in lines]
        return objects

    def get_data(self,idx):
        self.get_calibration(idx)
        return self.get_lidar(idx),self.get_label_objects(idx),self.get_image(idx)

class Object3d(object):
    """
    class as a storage of 3d bbox info
    """

    def __init__(self, label_file_line):
        # get one line in label file
        data = label_file_line.split(' ')
        data[1:] = [np.array(x,dtype=np.float32) for x in data[1:]]

        # data format
        # [type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y ,score(only for results)]
        # extract label, truncation, occlusion
        self.type = data[0]  # type(str)\: 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # occluded(int) :0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # alpha :object observation angle [-pi..pi]

        # extract 2d bounding box (image bounding box) in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box (3dpc bounding box) information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = np.array(data[11:14])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # rotation_y: yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        # print the info of target bounding box
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))