import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph.opengl as gl
from kitti_utils.read_kitti import Object3d
from kitti_utils.calibration import Calibration
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R
from pyqtgraph.Qt import QtCore, QtGui

import sys



class plot3dpc(object):
    def __init__(self,title='3D visualization'):
        # self.app = pg.mkQApp()
        self.app = QtGui.QApplication([])
        self.view = gl.GLViewWidget()
        self.view.resize(1920, 1080)
        self.coord = gl.GLAxisItem()
        self.coord.setSize(5, 5, 5)
        self.figure, self.ax = plt.subplots()
        # glLineWidth(3)
        self.view.addItem(self.coord)
        self.grid = gl.GLGridItem()
        self.view.addItem(self.grid)
        # self.view.setWindowTitle('3D visualization')
        self.points_items = []
        self.box_items = []
        self.orientation_items = []
        self.title = title

    def set_img(self, img):
        self.img = img

    def show(self,max_fps=10):
        self.view.setWindowTitle(self.title)
        self.view.show()
        if max_fps is not None:
            t = QtCore.QTimer()
            t.timeout.connect(self.update)
            t.start(1000/max_fps)
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    def show_img(self):
        if hasattr(self, 'img'):
            self.ax.imshow(self.img)
            plt.show()
            plt.close()
    def add_points(self, pc, color, size=3):
        """
        Add points to view
        :param pc: point cloud (N*4)
        :param color: color of point cloud (N*4)
        :param size: size of displayed points
        """
        self.sensor_num = 1
        self.time_step = 0
        self.total_time_length= 1
        points_item = gl.GLScatterPlotItem(pos=pc, color=color, size=size)
        self.view.addItem(points_item)
        self.points_items.append(points_item)
        return points_item
    def add_sequence_points(self,multisensor_sequence_pc, multisensor_sequence_color, size=3):
        """
        Add sequence points to view
        :param pc: point cloud (list [Num_LiDAR(RADAR), time , points])
        :param color: color of point cloud (list)
        :param size: size of displayed points
        """
        self.sequence_points_item = []
        self.sequence_points_color = []
        self.sensor_num = len(multisensor_sequence_pc)
        self.points_size = size
        for i in range(self.sensor_num):
            scatter_view = gl.GLScatterPlotItem(pos=np.zeros([1,3]), color=np.zeros([1,4]), size=size)
            self.view.addItem(scatter_view)
            self.points_items.append(scatter_view)
            self.sequence_points_item.append(multisensor_sequence_pc[i])
            self.sequence_points_color.append(multisensor_sequence_color[i])
            self.points_items[i].setData(pos=self.sequence_points_item[i][0],
                                         color=self.sequence_points_color[i][0], size=self.points_size)
        self.point_size = size
        self.time_step = 1
        self.total_time_length = len(self.sequence_points_item[0])

    def add_sequence_boxes(self,sequence_corners, sequence_ori=None):
        """
        # TODO
        Add sequence points to view
        :param sequence_corners: box corners (list [Num_LiDAR(RADAR), time , corners])
        :param sequence_ori: box orientations
        """
        self.sequence_corner = sequence_corners
        current_stacked_corners = self.sequence_corner[0]
        if sequence_ori is not None:
            self.sequence_ori = sequence_ori
            current_stacked_ori = self.sequence_ori[0]
            self.draw_3d_bbox(current_stacked_corners, current_stacked_ori)
        else:
            self.draw_3d_bbox(current_stacked_corners)


    def update(self):
        if getattr(self,'sequence_points_item',None) is not None:
            for i in range(self.sensor_num):
                self.points_items[i].setData(pos = self.sequence_points_item[i][self.time_step],color = self.sequence_points_color[i][self.time_step],size = self.points_size)
        if getattr(self,'sequence_corner',None) is not None:
            for box_item in self.box_items:
                self.view.removeItem(box_item)
            for ori_item in self.orientation_items:
                self.view.removeItem(ori_item)
            self.box_items = []
            self.orientation_items = []
            current_stacked_corners = self.sequence_corner[self.time_step]
            if getattr(self,'sequence_ori',None) is not None:
                current_stacked_ori = self.sequence_ori[self.time_step]
                self.draw_3d_bbox(current_stacked_corners,current_stacked_ori)
            else:
                self.draw_3d_bbox(current_stacked_corners)
        self.time_step +=1
        self.time_step = self.time_step%self.total_time_length



    def add_line(self, point1, point2, color=(1, 0, 0, 1)):
        """
        Draw a line

        :param point1: Vertex 1 of line
        :param point2: Vertex 1 of line
        """
        lines = np.stack([point1, point2], axis=0)
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=color, width=3, antialias=True)
        self.view.addItem(lines_item)
        return lines_item

    def add_box(self,vertexes ,color=(1, 0, 0, 1)):
        '''
        Draw a box with lines between each pair of vertexes

        :param vertexes: (8,3) array of floats specifying vertexes locations
        '''

        order_list = [0,1,0,2,3,1,3,2,4,5,4,6,7,5,7,6,0,4,1,5,2,6,3,7,2,7,3,6]
        corners = vertexes[order_list,:]
        box_item = gl.GLLinePlotItem(pos=corners, mode='lines',
                                       color=color, width=3, antialias=True)
        self.view.addItem(box_item)
        return box_item

    def show_img(self,title='3D visualization'):
        if hasattr(self, 'img'):
            self.ax.imshow(self.img)
            plt.show()
            plt.close()

    def exec(self):
        self.app.exec()

    def get_color(self, pc_intensity):
        """
        Get points' color by intensity
        :param pc: point cloud (N*1)
        :return: points' color (N*4)
        """
        imax, imin = np.max(pc_intensity), np.min(pc_intensity)
        color_ratio = 2 * (pc_intensity - imin) / (imax - imin)
        colors = np.zeros([pc_intensity.shape[0], 4])
        colors[:, 2] = np.maximum(1 - color_ratio, 0)
        colors[:, 0] = np.maximum(color_ratio - 1, 0)
        colors[:, 1] = 1 - colors[:, 1] - colors[:, 2]
        colors[:, 3] = np.ones([pc_intensity.shape[0]])
        return colors

    def covert_3dbox_corner(self, box3d, mode='center'):
        """
        Convert xyzhwl to 8 corners coord
        :param ob3ds: np.array([N*7]),c at center of the box
        :return: 8 corners coord in rect coord system
        """
        corners = np.zeros([len(box3d),8,3])
        orientation = np.zeros([len(box3d), 2, 3])
        ob3d = box3d

        c = ob3d[:,:3]
        w = ob3d[:,4]
        l = ob3d[:,3]
        h = ob3d[:,5]
        rz = ob3d[:,6]

        r = R.from_euler('xyz', np.concatenate([np.zeros_like(rz)[:,None],np.zeros_like(rz)[:,None],rz[:,None]],axis = 1))
        r1 = r.as_matrix()

        orientation[:,1, 0] = l
        orientation_3d = np.matmul(r1, orientation.swapaxes(1,2)).swapaxes(1,2)
        orientation_3d = orientation_3d + c[:,None,:]

        for ii in range(8):
            if ii & 1:
                corners[:,ii, 1] = c[:,1] + w / 2.0
            else:
                corners[:,ii, 1] = c[:,1] + -w / 2.0
            if ii & 2:
                corners[:,ii, 0] = c[:,0] + l / 2.0
            else:
                corners[:,ii, 0] = c[:,0] + -l / 2.0
            if ii & 4:
                if mode == 'center':
                    corners[:, ii, 2] = c[:, 2] + h / 2.0
                elif mode == 'bottom':
                    corners[:, ii, 2] = c[:, 2] + h
            else:
                if mode == 'center':
                    corners[:, ii, 2] = c[:, 2] + -h / 2.0
                elif mode == 'bottom':
                    corners[:, ii, 2] = c[:, 2] + 0

        corners = np.matmul(r1, (corners - c[:,None,:]).swapaxes(1,2)).swapaxes(1,2) + c[:,None,:]
        return corners, orientation_3d

    def draw_3d_bbox(self,stacked_corners, stacked_orientation_3d=None, color = (1,0,0,1)):
        for corners in stacked_corners:
            box = self.add_box(corners, color=color)
            self.box_items.append(box)
        if stacked_orientation_3d is not None:
            for orientation in stacked_orientation_3d:
                ori = self.add_line(orientation[0, :], orientation[1, :],color=color)
                self.orientation_items.append(ori)
        pass


    ##==================================================================================================================
    ## Only for KITTI, old code
    ##==================================================================================================================
    def draw_3d_bbox_kitti(self, calib: Calibration, corners_list, orientation_list):
        for corners in corners_list:
            corners = calib.project_rect_to_velo(corners)
            self.box_items.append(self.add_box(corners))
        for orientation in orientation_list:
            orientation = calib.project_rect_to_velo(orientation)
            self.orientation_items.append(self.add_line(orientation[0, :], orientation[1, :]))
        pass

    def draw_2d_bbox_kitti(self, calib: Calibration, corners_list):
        for corners in corners_list:
            corners = calib.project_rect_to_image(corners)
            self.ax.add_line(Line2D(corners[[0, 1], 0], corners[[0, 1], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[0, 2], 0], corners[[0, 2], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[3, 1], 0], corners[[3, 1], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[3, 2], 0], corners[[3, 2], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[4, 5], 0], corners[[4, 5], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[4, 6], 0], corners[[4, 6], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[7, 5], 0], corners[[7, 5], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[7, 6], 0], corners[[7, 6], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[0, 4], 0], corners[[0, 4], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[1, 5], 0], corners[[1, 5], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[2, 6], 0], corners[[2, 6], 1], linewidth=2, color='blue'))
            self.ax.add_line(Line2D(corners[[3, 7], 0], corners[[3, 7], 1], linewidth=2, color='blue'))
        pass

    def covert_3dbox_corner_kitti(self, ob3ds: Object3d):
        """
        Convert xyzhwl to 8 corners coord
        :param ob3ds: Object3d
        :return: 8 corners coord in rect coord system
        """
        corners_list = []
        orientation_3d_list = []
        for ob3d in ob3ds:

            if ob3d.type == 'DontCare': continue

            c = ob3d.t
            w = ob3d.w
            l = ob3d.l
            h = ob3d.h

            r = R.from_euler('xyz', [0, ob3d.ry, 0])
            r1 = r.as_matrix()

            orientation_3d = np.zeros([2, 3])
            orientation_3d[1, 0] = ob3d.l
            orientation_3d[[0,1], 1] = -ob3d.h/2.0
            # rotate and translate in camera coordinate system, project in image
            orientation_3d = np.matmul(r1, orientation_3d.T).T
            orientation_3d = orientation_3d + c
            orientation_3d_list.append(orientation_3d)

            corners = np.zeros([8, 3])

            for ii in range(8):
                if ii & 1:
                    corners[ii, 2] = c[2] + w / 2.0
                else:
                    corners[ii, 2] = c[2] + -w / 2.0
                if ii & 2:
                    corners[ii, 0] = c[0] + l / 2.0
                else:
                    corners[ii, 0] = c[0] + -l / 2.0
                if ii & 4:
                    corners[ii, 1] = c[1] + 0
                else:
                    corners[ii, 1] = c[1] + -h / 1.0

            corners = np.matmul(r1, (corners - c).T).T + c
            corners_list.append(corners)
        return corners_list, orientation_3d_list
