
from read_kitti import kitti_data
from visualization_module import plot3dpc
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', type=str, default='E:\DataSet\KITTI_LIDAR', help='Path of KITTI dataset')
    parser.add_argument(
        '--file_index', type=int, default=0, help='Which file you want to display')

    opt = parser.parse_args()
    data = kitti_data(opt.dataset_path)
    points, object_bboxs, image = data.get_data(opt.file_index)
    point_intensity = points[:, 3]
    points = points[:, :3]

    view = plot3dpc()
    view.add_point(pc=points, color=view.get_color(point_intensity))
    view.set_img(image)
    box_3dcorner_list, orientation_3d_list = view.covert_3dbox_corner(object_bboxs)
    view.draw_3d_bbox(calib=data.calib, corners_list=box_3dcorner_list, orientation_list=orientation_3d_list)
    view.draw_2d_bbox(calib=data.calib, corners_list=box_3dcorner_list)
    view.show()
if __name__ == '__main__':
    main()