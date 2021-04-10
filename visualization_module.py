import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from read_kitti import Object3d
from calibration import Calibration
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R


class plot3dpc(object):
    def __init__(self):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        self.view.resize(1920, 1080)
        self.coord = gl.GLAxisItem()
        self.coord.setSize(5, 5, 5)
        self.figure, self.ax = plt.subplots()
        glLineWidth(3)
        self.view.addItem(self.coord)

    def set_img(self, img):
        self.img = img

    def add_point(self, pc, color, size=3):
        """
        Add points to view
        :param pc: point cloud (N*4)
        :param color: color of point cloud (N*4)
        :param size: size of displayed points
        """
        points = gl.GLScatterPlotItem(pos=pc, color=color, size=size)
        self.view.addItem(points)

    def add_line(self, point1, point2):
        """
        Draw a line

        :param point1: Vertex 1 of line
        :param point2: Vertex 1 of line
        """
        lines = np.stack([point1, point2], axis=0)
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=(1, 0, 0, 1), width=3, antialias=True)
        self.view.addItem(lines_item)

    def show(self):
        if hasattr(self, 'img'):
            self.ax.imshow(self.img)
            plt.show()
            plt.close()
        self.view.show()
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
        colors[:, 1] = np.maximum(1 - color_ratio, 0)
        colors[:, 0] = np.maximum(color_ratio - 1, 0)
        colors[:, 2] = 1 - colors[:, 1] - colors[:, 2]
        colors[:, 3] = np.ones([pc_intensity.shape[0]])
        return colors

    def covert_3dbox_corner(self, ob3ds: Object3d):
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

    def draw_3d_bbox(self, calib: Calibration, corners_list, orientation_list):
        for corners in corners_list:
            corners = calib.project_rect_to_velo(corners)
            self.add_line(corners[0, :], corners[1, :])
            self.add_line(corners[0, :], corners[2, :])
            self.add_line(corners[3, :], corners[1, :])
            self.add_line(corners[3, :], corners[2, :])
            self.add_line(corners[4, :], corners[5, :])
            self.add_line(corners[4, :], corners[6, :])
            self.add_line(corners[7, :], corners[5, :])
            self.add_line(corners[7, :], corners[6, :])
            self.add_line(corners[0, :], corners[4, :])
            self.add_line(corners[1, :], corners[5, :])
            self.add_line(corners[2, :], corners[6, :])
            self.add_line(corners[3, :], corners[7, :])
        for orientation in orientation_list:
            orientation = calib.project_rect_to_velo(orientation)
            self.add_line(orientation[0, :], orientation[1, :])
        pass

    def draw_2d_bbox(self, calib: Calibration, corners_list):
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
