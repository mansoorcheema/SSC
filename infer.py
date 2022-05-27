from utils.seed import seed_torch
import os

import torch
import argparse
import numpy as np
from pathlib import Path
import imageio
import glob

from tqdm import tqdm
from torch.autograd import Variable
import datetime
from models import make_model
import config
import time

import VoxelUtils as vu

parser = argparse.ArgumentParser(description='PyTorch SSC Inference')
parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'nyucad', 'debug'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='palnet', choices=['ddrnet', 'aicnet', 'grfnet', 'palnet'],
                    help='model name (default: palnet)')
parser.add_argument('--files', default="/home/mcheem/data/datasets/large_room/", help='Depth Images')

parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--model_name', default='SSC_debug', type=str, help='name of model to save check points')


global args
args = parser.parse_args()

seed_torch(2021)

def _get_xyz(size):
        """x width yheight  zdepth"""
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h                 # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w                 # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d                 # z, front-back flip
        return _x, _y, _z


def labeled_voxel2ply(vox_labeled, ply_filename):  #
    """Save labeled voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
        vox_labeled.shape: (W, H, D)
    """  #
    # ---- Check data type, numpy ndarray
    if type(vox_labeled) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
    # ---- Check data validation
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    # ---- get size
    size = vox_labeled.shape
    # print('vox_labeled.shape:', vox_labeled.shape)
    # ---- Convert to list
    vox_labeled = vox_labeled.flatten()
    # ---- Get X Y Z
    _x, _y, _z = _get_xyz(size)
    _x = _x.flatten()
    _y = _y.flatten()
    _z = _z.flatten()
    # print('_x.shape', _x.shape)
    # ---- Get R G B
    vox_labeled[vox_labeled == 255] = 0  # empty
    # vox_labeled[vox_labeled == 255] = 12  # ignore
    _rgb = config.colorMap[vox_labeled[:]]
    # print('_rgb.shape:', _rgb.shape)
    # ---- Get X Y Z R G B
    xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # python2.7
    xyz_rgb = list(xyz_rgb)  # python3
    # print('xyz_rgb.shape-1', xyz_rgb.shape)
    # xyz_rgb = zip(_z, _y, _x, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # 将X轴和Z轴交换，用于meshlab显示
    # ---- Get ply data without empty voxel

    xyz_rgb = np.array(xyz_rgb)
    # print('xyz_rgb.shape-1', xyz_rgb.shape)
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]

    if len(ply_data) == 0:
        raise Exception("Oops!  That was no valid ply data.")
    ply_head = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header' % len(ply_data)
    # ---- Save ply data to disk
    np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
    del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head


def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
        r"""downsample the labeled data,
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        ds = downscale
        small_size = (voxel_size[0] // ds, voxel_size[1] // ds, voxel_size[2] // ds)  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)

        for i in range(small_size[0]*small_size[1]*small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            # z, y, x = np.unravel_index(i, small_size)  # 速度更慢了
            # print(x, y, z)

            label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            label_bin = label_i.flatten()  # faltten 返回的是真实的数组，需要分配新的内存空间
            # label_bin = label_i.ravel()  # 将多维数组变成 1维数组，而ravel 返回的是数组的视图

            # zero_count_0 = np.sum(label_bin == 0)
            # zero_count_255 = np.sum(label_bin == 255)
            zero_count_0 = np.array(np.where(label_bin == 0)).size  # 要比sum更快
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
                label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale

def _save_point_cloud(points, filename):
        with open(filename, "w") as f:
            for point in points.reshape(-1,3):
                f.write("{};{};{}\n".format(float(point[0]),float(point[1]), float(point[2])))
        

def _depth2voxel(depth, cam_pose, vox_origin, cam_k, voxel_unit=0.02, voxel_size = (240, 144, 240), ):
        #cam_k = param['cam_k']
        #voxel_size = (240, 144, 240)
        #unit = 0.02
        # ---- Get point in camera coordinate
        H, W = depth.shape
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter

        #_save_point_cloud(pt_cam, "point_cloud_cam.txt")
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        
        
        #vox_origin = pt_world.min(axis=(0,1))
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
        
        #_save_point_cloud(pt_world, "point_cloud_world.txt")
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        #pt_world2 = pt_world
        # pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        # pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 原始paper方法
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z
        _save_point_cloud(pt_world2, "point_cloud_world_aligned.txt")

        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / voxel_unit  # Get point in grid coordinate, each grid is a voxel

        _save_point_cloud(point_grid, "point_cloud_world_grid_aligned.txt")
        point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_size, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros([v for v in voxel_size], dtype=int)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_size + (3,), dtype=np.float32)  # (W, H, D, 3)
        position = np.zeros((H, W), dtype=np.int32)
        position4 = np.zeros((H, W), dtype=np.int32)
        # position44 = np.zeros((H/4, W/4), dtype=np.int32)

     
        voxel_size_lr = (voxel_size[0] // 4, voxel_size[1] // 4, voxel_size[2] // 4)
        for h in range(H):
            for w in range(W):
                i_x, i_y, i_z = point_grid[h, w, :]
                if 0 <= i_x < voxel_size[0] and 0 <= i_y < voxel_size[1] and 0 <= i_z < voxel_size[2]:
                    voxel_binary[i_x, i_y, i_z] = 11  # the bin has at least one point (bin is not empty)
                    voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                    # position[h, w, :] = point_grid[h, w, :]  # 记录图片上的每个像素对应的voxel位置
                    # 记录图片上的每个像素对应的voxel位置
                    position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_size)
                    # TODO 这个project的方式可以改进
                    position4[h, ] = np.ravel_multi_index((point_grid[h, w, :] / 4).astype(np.int32), voxel_size_lr)
                    # position44[h / 4, w / 4] = np.ravel_multi_index(point_grid[h, w, :] / 4, voxel_size_lr)

        # output --- 3D Tensor, 240 x 144 x 240

        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid  # Release Memory
        return voxel_binary, voxel_xyz, position, position4  # (W, H, D), (W, H, D, 3)


def compute_tsdf(depth_data, vox_origin, cam_k, cam_pose0, voxel_size=(240,144,240)):
    # cam_info_CPU, vox_info_CPU,depth_data_CPU, vox_tsdf_CPU, depth_mapping_idxs_CPU
    height, width = depth_data.shape
    vox_tsdf = np.ones(voxel_size[0] * voxel_size[1] * voxel_size[2], dtype=np.float64)
    depth_mapping_idxs = np.zeros(width * height, dtype=np.float64)
    voxel_occupancy = np.zeros(voxel_size[0] * voxel_size[1] * voxel_size[2], dtype=np.float64 )

    # setup camera info
    cam_info = np.zeros(27,dtype=np.float64)
    cam_info[0] = width
    cam_info[1] =  height

    for i in range(9):
        cam_info[i + 2] = np.asarray(cam_k).reshape(-1)[i]
  
    for i in range(16):
        cam_info[i + 11] = cam_pose0.reshape(-1)[i]

    # setup voxel info
    vox_info = np.zeros(8,dtype=np.float64)
    vox_info[0] = 0.02; # vox unit
    vox_info[1] = 0.04; # vox margin

    for i in range(3):
        vox_info[i + 2] = voxel_size[i]

    for i in range(3):
        vox_info[i + 5] = vox_origin[i]

    depth_data_reshaped = depth_data.astype(np.float64).reshape(-1)

    vu.compute_tsdf(cam_info, vox_info, depth_data_reshaped, vox_tsdf, depth_mapping_idxs, voxel_occupancy) 
    #print (vox_tsdf_CPU.shape)
    return vox_tsdf.reshape(voxel_size), depth_mapping_idxs.reshape( height, width), \
         voxel_occupancy.reshape(voxel_size).astype(np.int64) * 11

def get_origin_from_depth_image(depth, cam_k, cam_pose):
    """
    Get Point cloud origin in world coordinates
    """
    #cam_k = param['cam_k']
    #voxel_size = (240, 144, 240)
    #unit = 0.02
    # ---- Get point in camera coordinate
    H, W = depth.shape
    gx, gy = np.meshgrid(range(W), range(H))
    pt_cam = np.zeros((H, W, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter

    #_save_point_cloud(pt_cam, "point_cloud_cam.txt")
    # ---- Get point in world coordinate
    p = cam_pose
    pt_world = np.zeros((H, W, 3), dtype=np.float32)
    pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
    pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
    pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
    
    
    vox_origin = pt_world.min(axis=(0,1))
    return vox_origin

def load_data_from_depth_image(filename):
    #filename0="/home/mcheem/data/datasets/large_room/frame_541.png"
    #filename="/Data/Study/ETH/Thesis/code/Utils/depth_from_rosbag/depth_images_app_sync/000541_depth.npz"
    #depth = imageio.imread(filename)
    #depth = np.asarray(depth/ 8000.0)  # numpy.float64
    # assert depth.shape == (img_h, img_w), 'incorrect default size'
    #depth = np.asarray(depth)
    #depth = np.asarray(depth)
    rgb = None
    frame_data = np.load(filename[:-4] + ".npz")
    depth_npy = frame_data["depth"]
    cam_pose0 = frame_data["pose"]
    

    #depth[depth>8] = depth.min()
    #depth_npy[depth_npy>8] = depth_npy.min()
    # with open(filename[:-10] + "pose.txt") as f:
    #     lines = f.read().splitlines()

    # cam_pose = np.asarray([np.fromstring(l, dtype=np.float32, sep=' ') for l in lines])
    
    #this is the camera intrinsic that we have been using all along
    # cam_k = [[320, 0, 320],  
    #         [0, 320, 240],  
    #         [0, 0, 1]]
    
    # this if from the ros bag that lukas shared
    cam_k = [[504.6463623046875, 0.0, 330.97552490234375], [0.0, 504.8226318359375, 327.1402282714844], [0.0, 0.0, 1.0]]
    #vox_origin = [0,0,0,0] #[-2.75309,0.931033,-0.05]
    vox_origin = get_origin_from_depth_image(depth_npy,cam_k, cam_pose0) 
    # cam_info_CPU, vox_info_CPU,depth_data_CPU, vox_tsdf_CPU, depth_mapping_idxs_CPU
    
    #vox_tsdf, depth_mapping_idxs, voxel_occupancy = compute_tsdf(depth_npy, vox_origin, cam_k, cam_pose0)
   
     
    #cam_pose0 = np.array([[1.0,0,0,0],[0,0,1.0,0],[0,-1.0,0,0],[0,0,0,1]]).dot(cam_pose0)
    voxel_binary, voxel_xyz, position, position4 = _depth2voxel(depth_npy, cam_pose0, vox_origin, cam_k)
    #voxel_binary = _downsample_label(voxel_binary)
    #voxel_occupancy = _downsample_label(voxel_occupancy)
    #labeled_voxel2ply(voxel_occupancy,"scan_occ2.ply")
    #labeled_voxel2ply(voxel_binary,"scan_occ.ply")
    #np.set_printoptions(suppress=True)
    #print(cam_pose)
    #print (cam_pose0)
    return rgb, depth_npy, None, position, voxel_binary
    #return rgb, depth_npy, None, depth_mapping_idxs, voxel_occupancy.transpose(2,1,0)
    #return rgb, torch.as_tensor(depth_npy).unsqueeze(0).unsqueeze(0), torch.as_tensor(vox_tsdf).unsqueeze(0), torch.as_tensor(depth_mapping_idxs).unsqueeze(0).unsqueeze(0), torch.as_tensor(voxel_occupancy.transpose(2,1,0)).unsqueeze(0)

def load_occupancy_prob_from_depth_image(filename, max_depth=8, cam_k=[[320, 0, 320], [0, 320, 240], [0, 0, 1]]):
    """
    Read depth and pose froms ave npz file and return tsdf voxels.
    """
    rgb = None
    frame_data = np.load(filename[:-4] + ".npz")
    depth_npy = frame_data["depth"]
    cam_pose = frame_data["pose"]

    depth_npy[depth_npy > max_depth] = depth_npy.min()
    vox_origin = utils.get_origin_from_depth_image(depth_npy, cam_k, cam_pose)

    _, _, voxel_occupancy = utils.compute_tsdf(
        depth_npy, vox_origin, cam_k, cam_pose, voxel_size=(64,64,64), voxel_unit_size=0.08)
    voxel_occupancy_tensor =  torch.as_tensor(voxel_occupancy.transpose(2, 1, 0)).unsqueeze(0) / 11.0
    #log_odds_occupancy = voxel_occupancy_tensor * np.log(0.9/0.1)
    return voxel_occupancy_tensor# log_odds_occupancy

def infer():
    net = make_model(args.model, num_classes=12)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            cp_states = torch.load(args.resume, map_location=torch.device('cpu'))
            net.load_state_dict(cp_states['state_dict'], strict=False)
        else:
            raise Exception("=> NO checkpoint found at '{}'".format(args.resume))


    net.eval()  # switch to train mode
    torch.cuda.empty_cache()
    
    #file_list = os.listdir(args.files)
    file_list = glob.glob(str(Path(args.files) / "*.npz"))
    # scene_inpainting_network = torch.jit.load("/Data/SI_ScanNet_0614.pt",torch.device('cpu'))

    # t = torch.randn(1,64,64,64).float()
    # m = torch.ones(1,64,64,64).float()
    # tt = 0
    # x_depth= torch.randn(1,1,640,480).float()
    # x_tsdf=torch.randn(1,240,144,240).float()
    # p = torch.ones(640,480).long()
    # for i in range(10):
    #     time1 = time.time()
    #     #y_pred = scene_inpainting_network.forward(t, m)
    #     y_pred = net(x_depth=x_depth, x_tsdf=x_tsdf, p=p)
    #     time2 = time.time()
    #     tt += (time2-time1)*1000.0
    # tt /= 10

    # for name, para in scene_inpainting_network.named_parameters():
    #     print('{}: {}'.format(name, para.shape))

    for step, depth_file in enumerate(file_list):
        rgb, depth, tsdf, position, occupancy_grid = load_data_from_depth_image(depth_file)
        
        ####################
        # occ_map = load_occupancy_prob_from_depth_image(depth_file)
        # log_odds_map = occ_map * np.log(0.8/0.2)
        # mask = log_odds_map <= 0
        #inputs= torch.tensor([log_odds_map.float(),mask.float()])
        #y_pred = scene_inpainting_network.forward(log_odds_map.float())
        # ---- (bs, C, D, H, W), channel first for Conv3d in pyTorch
        # FloatTensor to Variable. (bs, channels, 240L, 144L, 240L)
        x_depth = Variable(torch.tensor(depth).unsqueeze(0).unsqueeze(0))
        position = torch.tensor(position).unsqueeze(0).unsqueeze(0).long()

        if args.model == 'palnet':
            x_tsdf = Variable(torch.zeros(occupancy_grid.shape).float().unsqueeze(0))
            y_pred = net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
        else:
            x_rgb = Variable(rgb.float())
            y_pred = net(x_depth=x_depth, x_rgb=x_rgb, p=position)

        
        scores = torch.nn.Softmax(dim=0)(y_pred.view(1,12,-1)[0])
        scores[0] += 0.3 #Increase offset of empty class to weed out low prob predictions
        pred_cls = torch.argmax(scores, dim=0)

        pred_cls = pred_cls.reshape(1,60,36,60)

        #labeled_voxel2ply(target[0].numpy(),"outputs_thresh/target{}.ply".format(step))
        labeled_voxel2ply(pred_cls[0].cpu().numpy(),"outputs/{}_preds.ply".format(Path(depth_file).stem))
        occupancy_grid_downsampled = _downsample_label(occupancy_grid)
        labeled_voxel2ply(occupancy_grid_downsampled,"outputs/{}_scan.ply".format(Path(depth_file).stem))
        # pass
       

def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("CUDA device found!".format(torch.cuda.device_count()))
    else:
        print("Using CPU!")

    infer()


if __name__ == '__main__':
    main()
