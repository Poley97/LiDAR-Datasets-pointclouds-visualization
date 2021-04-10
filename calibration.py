import numpy as np
import os


class Calibration(object):
    """
    Reference : Geiger, Andreas, et al. "Vision meets robotics: The kitti dataset."
    The International Journal of Robotics Research 32.11 (2013): 1231-1237.
    """

    def __init__(self, dataset_path, split='training'):
        self.dataset_path = dataset_path
        self.split = split
        self.index = -1
        self.data_file = ''
        self.calib = {}
        pass

    def read_cal_file(self, index):
        """
        Read calibration file info

        :param index: calibration file index
        """
        self.data_file = str('%.6d.txt' % index)
        data_file_path = os.path.join(self.dataset_path, self.split, 'calib', self.data_file)
        f = open(data_file_path)
        for line in f.readlines():
            if line == '\n':
                continue
            name, data = line.split(': ')
            data = np.array(data.strip('\n').split(' '), dtype=np.float)
            # print(name,' : ',data.shape[0])
            if data.shape[0] == 9:
                data = data.reshape([3, 3])
            elif data.shape[0] == 12:
                data = data.reshape([3, 4])
            self.calib[name] = data
        pass

    def get_P(self):
        """
        Get P2 matrix

        :return: Projection matrix P2
        """
        assert 'P2' in self.calib, 'No calibration file has been read !'
        return self.calib['P2']

    def get_V2C(self):
        """
        Get Tr_velo_to_cam matrix

        :return: Projection matrix from velo to cam Tr_velo_to_cam
        """
        assert 'Tr_velo_to_cam' in self.calib, 'No calibration file has been read !'
        return self.calib['Tr_velo_to_cam']

    def get_I2V(self):
        """
        Get Tr_imu_to_velo matrix

        :return: Projection matrix from IMU to velo Tr_imu_to_velo
        """
        assert 'Tr_imu_to_velo' in self.calib, 'No calibration file has been read !'
        return self.calib['Tr_imu_to_velo']

    def get_R0(self):
        """
        Get R0_rect matrix

        :return: Rotation matrix R0_rect
        """
        assert 'R0_rect' in self.calib, 'No calibration file has been read !'
        return self.calib['R0_rect']

    def expand_R0(self,R0):
        """
        Expand R0 from 3*3 to 4*4

        :param R0:Rotation matrix 3*3
        :return:Expanded Rotation matrix 4*4
        """
        eR0=np.zeros([4,4])
        eR0[0:3,0:3]=R0
        eR0[3,3]=1
        return eR0
    def current_cal_file_index(self):
        """
        :return:  File index read by object now
        """
        return self.index

    def inverse_rigid_trans(self, Tr):
        """
        Inverse of rigid body transformation matrix

        :param Tr: rigid body transformation matrix
        :return:iTr : inverse Tr
        """
        iTr = np.zeros_like(Tr)  # 3x4
        iTr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        iTr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return iTr

    def cart2hom(self, pts_3d):
        """
        Transform coordinates from Cartesian coordinate system to homogeneous coordinate system

        :param pts_3d:Coordinates in Cartesian
        :return:Coordinates in homogeneous
        """
        N = pts_3d.shape[0]
        pts_3d_hom = np.concatenate([pts_3d, np.ones((N, 1))], axis=1)
        return pts_3d_hom

    def get_camera_intrinsics_and_extrinsics(self,P):
        """
        Get camera intrinsic and extrinsic

        :P:Projection matrix
        :return: camera intrinsics and extrinsics
        """
        fx = P[0, 0]
        fy = P[1, 1]
        cx = P[0, 2]
        cy = P[1, 2]
        tx = P[0, 3] / (-fx)
        ty = P[1, 3] / (-fy)
        return fx,fy,cx,cy,tx,ty
    # ==================================================================================================================
    # =================================== Transformation between 3D and 3D =============================================
    # ==================================================================================================================
    def project_velo_to_ref(self, pts_3d_velo):
        """
        Coordinates transformation between velo and ref

        :param pts_3d_velo: Coordinates in velo coordinate system
        :return: Coordinates in ref coordinate system
        """
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.matmul(self.get_V2C(), pts_3d_velo.T).T

    def project_ref_to_velo(self, pts_3d_ref):
        """
        Coordinates transformation between ref and velo

        :param pts_3d_ref: Coordinates in ref coordinate system
        :return: Coordinates in velo coordinate system
        """
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.matmul(self.inverse_rigid_trans(self.get_V2C()), pts_3d_ref.T).T

    def project_rect_to_ref(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        camera 0 coordinate system

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in camera 0 coordinate system
        """
        return np.transpose(np.dot(np.linalg.inv(self.get_R0()), pts_3d_rect.T))

    def project_ref_to_rect(self, pts_3d_ref):
        """
        Coordinates transformation between camera 0 coordinate system and
        camera 2 coordinate system

        :param pts_3d_ref: Coordinates in camera 0 coordinate system
        :return: Coordinates in camera 2 coordinate system
        """
        return np.transpose(np.dot(self.get_R0(), np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        velo coordinate system

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in velo coordinate system
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        """
        Coordinates transformation between velo coordinate system and
        camera 2 coordinate system

        :param pts_3d_velo: Coordinates in velo coordinate system
        :return: Coordinates in camera 2 coordinate system
        """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ==================================================================================================================
    # =================================== Transformation between 3D and 2D =============================================
    # ==================================================================================================================
    def project_rect_to_image(self, pts_3d_rect):
        """
        Coordinates transformation between camera 2 coordinate system and
        image2

        :param pts_3d_rect: Coordinates in camera 2 coordinate system
        :return: Coordinates in image 2
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        # pts_3d_rect2 = np.matmul(self.expand_R0(self.get_R0()),pts_3d_rect.T).T
        # pts_2d = np.matmul(self.get_P(), pts_3d_rect2.T).T
        pts_2d = np.matmul(self.get_P(), pts_3d_rect.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """
        Coordinates transformation between velo coordinate system and
        image2

        :param pts_3d_rect: Coordinates in velo coordinate system
        :return: Coordinates in image 2
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    ''' Input: nx3 first two channels are uv, 3rd channel
                       is depth in rect camera coord.
                Output: nx3 points in rect camera coord.
            '''""
    def project_image_to_rect(self, xy_depth):
        """
        Coordinates transformation between image2 coordinate system and
        camera 0

        :param xy_depth: channel 1: image_x ; channel 2: image_y ; channel 3: depth
        :return: Coordinates in camera 0 coordinate system
        """

        fx,fy,cx,cy,tx,ty=self.get_camera_intrinsics_and_extrinsics(self.get_P())
        N = xy_depth.shape[0]
        x = ((xy_depth[:, 0] - cx) * xy_depth[:, 2]) / fx + tx
        y = ((xy_depth[:, 1] - cy) * xy_depth[:, 2]) / fy + ty
        pts_3d_rect = np.zeros((N, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = xy_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, xy_depth):
        """
        Coordinates transformation between image2 coordinate system and
        velo coordinate system

        :param xy_depth: channel 1: image_x ; channel 2: image_y ; channel 3: depth
        :return: Coordinates in velo coordinate system
        """
        pts_3d_rect = self.project_image_to_rect(xy_depth)
        return self.project_rect_to_velo(pts_3d_rect)


if __name__ == '__main__':
    C = Calibration('D:/Dataset/KITTI_LIDAR')
    C.read_cal_file(1)
    print(C.get_P())
