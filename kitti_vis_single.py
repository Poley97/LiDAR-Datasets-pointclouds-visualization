from kitti_utils.read_kitti import kitti_data
from vis_utils.visualization_module_v1 import plot3dpc
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', type=str, default='D:\Datasets\kitti_lidar', help='Path of KITTI dataset')
    parser.add_argument(
        '--file_index', type=int, default=1, help='Which file you want to display')

    opt = parser.parse_args()
    data = kitti_data(opt.dataset_path)
    points, object_bboxs, image = data.get_data(opt.file_index)
    point_intensity = points[:, 3]
    points = points[:, :3]
    view = plot3dpc()


    view.add_points(pc=points, color=view.get_color(point_intensity))
    box_3dcorner_list, orientation_3d_list = view.covert_3dbox_corner_kitti(object_bboxs)
    view.draw_3d_bbox_kitti(data.calib,box_3dcorner_list, orientation_3d_list)
    view.show(max_fps=None)
    view.set_img(image)
    view.draw_2d_bbox_kitti(calib=data.calib, corners_list=box_3dcorner_list)
    view.show_img()
if __name__ == '__main__':
    main()