import imp
m = imp.find_module('waymo_open_dataset', ['.'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow.compat.v1 as tf
from waymo_utils import read_tfrecord
from vis_utils.visualization_module_v1 import plot3dpc
import argparse
import numpy as np
from waymo_utils.generate_labels import generate_labels


def waymo_vis(opt):
    """

    :param file_path: path of the waymo lidar data (tfrecord)
    """
    tf.disable_v2_behavior()
    tf.enable_eager_execution()
    FILENAME = opt.filepath
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    view = plot3dpc()
    start_frame = opt.start_frame
    end_frame = opt.end_frame
    sequence_points = [[] for i in range(5)] # 5 LiDAR in Waymo
    sequence_colors = [[] for i in range(5)]
    sequence_annotation = []
    for i, data in enumerate(dataset):
        if i >= start_frame and i <= end_frame:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            (range_images, camera_projections,
             range_image_top_pose) = read_tfrecord.parse_range_image_and_camera_projection(frame)
            sequence_annotation.append(generate_labels(frame)['gt_boxes_lidar'])
            points, cp_points = read_tfrecord.convert_range_image_to_point_cloud(frame,
                                                                                 range_images,
                                                                                 camera_projections,
                                                                                 range_image_top_pose)
            for j in range(5): # There are totally 5 LiDAR in waymo open dataset
                sequence_points[j].append(points[j][:,:3])
                sequence_colors[j].append(view.get_color(np.tanh(points[j][:, 3])+1))
        else:
            continue

    sequence_boxes_corner_list = []
    sequence_boxes_ori_list = []
    for i in range(len(sequence_annotation)):
        corners,ori = view.covert_3dbox_corner(sequence_annotation[i])
        sequence_boxes_corner_list.append(corners)
        sequence_boxes_ori_list.append(ori)
    view.add_sequence_boxes(sequence_boxes_corner_list,sequence_boxes_ori_list)
    view.add_sequence_points(sequence_points,sequence_colors)
    view.show(opt.maxfps)
    pass



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath', type=str, default='C:/Users/lenovo/Desktop/kitti-3d-visual/data/Waymo/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord',
            help='The data you want to do visualization')
    parser.add_argument(
        '--maxfps', type=int, default=10,
            help='The max fps during sequence visualization')
    parser.add_argument(
        '--start_frame', type=int, default=0,
            help='where the visualization started from')
    parser.add_argument(
        '--end_frame', type=int, default=50,
            help='where the visualization ended at')

    opt = parser.parse_args()
    waymo_vis(opt)