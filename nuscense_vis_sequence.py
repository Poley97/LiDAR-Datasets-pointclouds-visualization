
from vis_utils.visualization_module_v1 import plot3dpc
import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscense_utils.data_utils import get_sample_data,get_available_scenes
version = 'v1.0-trainval'



def nuScense_vis(opt):
    """

    :param file_path: path of the waymo lidar data (tfrecord)
    """
    nusc = NuScenes(version=version, dataroot=opt.rootpath, verbose=True)
    scene_token = nusc.scene[opt.scene_num]['token']
    scene_rec = nusc.get('scene', scene_token)
    sample_token = scene_rec['first_sample_token']
    sequence_points_list = []
    sequence_corners_list = []
    sequence_colors_list = []
    view = plot3dpc()
    corner_order = [6,7,2,3,5,4,1,0]
    max_frame = opt.max_frame
    frame_idx = 0
    while sample_token!='':
        if frame_idx >= max_frame:
            break
        sample_rec = nusc.get('sample', sample_token)
        lidar_token = sample_rec['data']['LIDAR_TOP']
        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, lidar_token)
        lidar_scan = np.fromfile(ref_lidar_path, dtype=np.float32).reshape((-1, 5))
        sequence_points_list.append(lidar_scan[:,:3])
        sequence_colors_list.append(view.get_color((lidar_scan[:,3])))
        sequence_corners_list.append(np.concatenate([ref_boxes[i].corners().transpose(1, 0)[None,corner_order,:] for i in range(len(ref_boxes))],axis=0))
        sample_token = sample_rec['next']
        frame_idx += 1
    view.add_sequence_boxes(sequence_corners_list)
    view.add_sequence_points([sequence_points_list],[sequence_colors_list])
    view.show(opt.maxfps)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rootpath', type=str, default='D:/Datasets/nuscenes_part/',
            help='The path of nuScense dataset ')
    parser.add_argument(
        '--maxfps', type=int, default=5,
            help='The max fps during sequence visualization')
    parser.add_argument(
        '--scene_num', type=int, default=0,
            help='which scene you want to see')
    parser.add_argument(
        '--max_frame', type=int, default=40,
            help='which scene you want to see')


    opt = parser.parse_args()
    nuScense_vis(opt)