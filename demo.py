from vis_utils.visualization_module_v1 import plot3dpc
import numpy as np
def main():

    pc_info = np.load('demo/demo_points.npy')
    target = np.load('demo/demo_boxes.npy', allow_pickle=True)[0]
    view = plot3dpc()
    stacked_corners, stacked_orientation = view.covert_3dbox_corner(target[0][:, :7],mode='bottom')
    view.draw_3d_bbox(stacked_corners, stacked_orientation, color=[0,1,0,1])
    view.add_points(pc=pc_info[:, :3], color=view.get_color(pc_info[:, 3]), size=5)
    view.show(max_fps=None)

if __name__=='__main__':
    main()