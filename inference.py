""" 
@Date: 2021/09/19
@description:
"""
import json
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import math
from tqdm import tqdm
from PIL import Image
from config.defaults import merge_from_file, get_config
from dataset.mp3d_dataset import MP3DDataset
from dataset.zind_dataset import ZindDataset
from models.build import build_model
from loss import GradLoss
from postprocessing.post_process import post_process
from preprocessing.pano_lsd_align import panoEdgeDetection, rotatePanorama
from utils.boundary import corners2boundaries, corners2boundary, layout2depth, visibility_corners, calc_rotation
from utils.conversion import depth2xyz, xyz2pixel, uv2xyz, xyz2depth, pixel2xyz
from utils.logger import get_logger
from utils.misc import tensor2np_d, tensor2np
from evaluation.accuracy import show_grad
from models.lgt_net import LGT_Net
from utils.writer import xyz2json
from visualization.boundary import draw_boundaries, draw_walls
from visualization.floorplan import draw_floorplan, draw_iou_floorplan
from visualization.obj3d import create_3d_obj
from visualization.grad import convert_img


def parse_option():
    parser = argparse.ArgumentParser(description='Panorama Layout Transformer training and evaluation script')
    parser.add_argument('--data_glob',
                        type=str,
                        required=True,
                        help='image glob path')

    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar='FILE',
                        help='path of config file')

    parser.add_argument('--post_processing',
                        type=str,
                        default='manhattan',
                        choices=['manhattan', 'atalanta', 'original'],
                        help='post-processing type')

    parser.add_argument('--output_dir',
                        type=str,
                        default='src/output',
                        help='path of output')

    parser.add_argument('--visualize_3d', action='store_true',
                        help='visualize_3d')

    parser.add_argument('--output_3d', action='store_true',
                        help='output_3d')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='device')

    args = parser.parse_args()
    args.mode = 'test'

    print("arguments:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("-" * 50)
    return args


def visualize_2d(img, corners, dt, show_depth=True, show_floorplan=True, show=False, save_path=None):
    #dt_np = tensor2np_d(dt)
    floor_xyz = uv2xyz(corners[len(corners)//2:])
    # dt_depth = dt_np['depth'][0]
    # dt_xyz = depth2xyz(np.abs(dt_depth))
    # dt_ratio = dt_np['ratio'][0][0]
    corners_list = []
    corners_list.append(corners[:len(corners)//2])
    corners_list.append(corners[len(corners)//2:])
    #dt_boundaries = corners2boundaries(ratio=1, corners_xyz=floor_xyz, step=None, visible=False, length=img.shape[1])
    vis_img = draw_boundaries(img,corners_list, boundary_color=[0, 1, 0])

    # if 'processed_xyz' in dt:
    #     dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][0], step=None, visible=False,
    #                                        length=img.shape[1])
    #     vis_img = draw_boundaries(vis_img, boundary_list=dt_boundaries, boundary_color=[1, 0, 0])

    if show_depth:
        dt_grad_img = show_depth_normal_grad(dt)
        grad_h = dt_grad_img.shape[0]
        vis_merge = [
            vis_img[0:-grad_h, :, :],
            dt_grad_img,
        ]
        vis_img = np.concatenate(vis_merge, axis=0)
        # vis_img = dt_grad_img.transpose(1, 2, 0)[100:]

    if show_floorplan:
        floorplan = show_alpha_floorplan(floor_xyz, border_color=[0, 1, 0, 1])

        vis_img = np.concatenate([vis_img, floorplan[:, 40:-40, :]], axis=1)
    if show:
        plt.imshow(vis_img)
        plt.show()
    if save_path:
        result = Image.fromarray((vis_img * 255).astype(np.uint8))
        result.save(save_path)
    return vis_img


def preprocess(img_ori, q_error=0.7, refine_iter=3, vp_cache_path=None):
    # Align images with VP
    if os.path.exists(vp_cache_path):
        with open(vp_cache_path) as f:
            vp = [[float(v) for v in line.rstrip().split(' ')] for line in f.readlines()]
            vp = np.array(vp)
    else:
        # VP detection and line segment extraction
        _, vp, _, _, _, _, _ = panoEdgeDetection(img_ori,
                                                 qError=q_error,
                                                 refineIter=refine_iter)
    i_img = rotatePanorama(img_ori, vp[2::-1])

    if vp_cache_path is not None:
        with open(vp_cache_path, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))

    return i_img, vp


def show_depth_normal_grad(dt):
    dt_grad_img = show_grad(dt['trivialWalls'][0], 10)
    dt_grad_img = cv2.resize(dt_grad_img, (1024, 40), interpolation=cv2.INTER_NEAREST)
    return dt_grad_img


def show_alpha_floorplan(dt_xyz, side_l=512, border_color=None):
    if border_color is None:
        border_color = [1, 0, 0, 1]
    fill_color = [0.2, 0.2, 0.2, 0.2]
    dt_floorplan = draw_floorplan(xz=dt_xyz[..., ::2], fill_color=fill_color,
                                  border_color=border_color, side_l=side_l, show=False, center_color=[1, 0, 0, 1])
    dt_floorplan = Image.fromarray((dt_floorplan * 255).astype(np.uint8), mode='RGBA')
    back = np.zeros([side_l, side_l, len(fill_color)], dtype=float)
    back[..., :] = [0.8, 0.8, 0.8, 1]
    back = Image.fromarray((back * 255).astype(np.uint8), mode='RGBA')
    iou_floorplan = Image.alpha_composite(back, dt_floorplan).convert("RGB")
    dt_floorplan = np.array(iou_floorplan) / 255.0
    return dt_floorplan


def save_pred_json(xyz, ration, save_path):
    # xyz[..., -1] = -xyz[..., -1]
    json_data = xyz2json(xyz, ration)
    with open(save_path, 'w') as f:
        f.write(json.dumps(json_data, indent=4) + '\n')
    return json_data

def save_trivialWalls(dt,save_path):
    with open(save_path,'w') as file:
        file.write(str(dt['trivialWalls'][0]))

def inference():
    if len(img_paths) == 0 :
        logger.error('No images found')
        return
    if len(corners_paths) == 0 :
        logger.error('No corners found')
        return
    assert len(img_paths) == len(corners_paths), "inputs pair match error"

    bar = tqdm(zip(img_paths,corners_paths), ncols=100)
    for img_path, corners_path in bar:
        if not os.path.isfile(img_path):
            logger.error('The {} not is file'.format(img_path))
            continue
        name = os.path.basename(img_path).split('.')[0]
        bar.set_description(name)
        img = np.array(Image.open(img_path).resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]

        # corners format:
        # pixel coordinate (512,1024)
        # [ceiling,floor]
        corners = np.loadtxt(corners_path)/ img.shape[:2][::-1]
        floor_uv = corners[len(corners)//2:]
        if args.post_processing is not None and 'manhattan' in args.post_processing:
            bar.set_description("Preprocessing")
            rotation = calc_rotation(floor_uv)
            shift = math.modf(rotation / (2 * np.pi) + 1)[0]
            img = np.roll(img, round(shift * img.shape[1]), axis=1)
            corners[:, 0] = np.modf(corners[:, 0] + shift)[0]

        img = (img / 255.0).astype(np.float32)
        run_one_inference(img, corners, model, args, name, logger)


def inference_dataset(dataset):
    bar = tqdm(dataset, ncols=100)
    for data in bar:
        bar.set_description(data['id'])
        run_one_inference(data['image'].transpose(1, 2, 0), model, args, name=data['id'], logger=logger)

# 舊版
# def cal_tw(x1,x2,dt,last_wall = False):
#     x1 = int(x1)
#     x2 = int(x2)
#     if not last_wall:
#         avg_tw = np.sum(dt[x1:x2])/(x2-x1)
#     else:
#         avg_tw = (np.sum(dt[:x1])+np.sum(dt[x2:]))/(x1+(256-x2))

#     return avg_tw

def cal_tw(x1, x2, dt, last_wall=False):
    try:
        x1 = int(x1)
        x2 = int(x2)

        if not isinstance(dt, np.ndarray):
            raise ValueError("Input dt must be a numpy array")

        if x1 < 0 or x2 < 0 or x1 > len(dt) or x2 > len(dt):
            raise IndexError("x1 and x2 must be within the bounds of dt")

        if x1 == x2:
            raise ValueError("x1 and x2 cannot be the same value")

        if not last_wall:
            avg_tw = np.sum(dt[x1:x2]) / (x2 - x1)
        else:
            if x1 == 0 and x2 == len(dt):
                raise ValueError("For the last wall case, x1 and x2 cannot be at the array bounds")

            avg_tw = (np.sum(dt[:x1]) + np.sum(dt[x2:])) / (x1 + (256 - x2))

        return avg_tw

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except IndexError as ie:
        print(f"IndexError: {ie}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def get_depth(corners, plan_y=1, length=256, visible=True):
    visible_floor_boundary = corners2boundary(corners, length=length, visible=visible)
    # The horizon-depth relative to plan_y
    visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, plan_y), plan_y)
    return visible_depth

def get_depth_map(img,corners):
    cei = corners[len(corners)//2:]
    floor = corners[:len(corners)//2]
    depth = get_depth(floor,length=img.shape[1],visible=True)
    depth_img = np.expand_dims(depth,axis=0) 
    depth_img = np.repeat(depth_img,img.shape[0],axis=0) 
    depth_img = np.expand_dims(depth_img,axis=-1)
    depth_img = np.repeat(depth_img,3,axis=-1)
    cor_list = []
    cor_list.append(cei)
    cor_list.append(floor)
    depth_img = draw_walls(depth_img,cor_list,ch_num=1)
    return depth_img

def get_normal_map(img, corners):
    cei = corners[len(corners)//2:]
    floor = corners[:len(corners)//2]
    depth = torch.tensor(get_depth(floor,length=img.shape[1],visible=True))
    xz = depth2xyz(depth)[:,::2]
    direction = torch.roll(xz, -1, dims=0) - xz  # direct[i] = xz[i+1] - xz[i]
    direction = direction / direction.norm(p=2, dim=-1)[..., None]
    angle = torch.atan2(direction[..., 1], direction[..., 0])
    normal_img = convert_img(angle, img.shape[0], cmap='HSV')
    cor_list = []
    cor_list.append(cei)
    cor_list.append(floor)
    normal_img = draw_walls(normal_img,cor_list)
    return normal_img


@torch.no_grad()
def run_one_inference(img, corners, model, args, name, logger, show=False, show_depth=True,
                      show_floorplan=True, mesh_format='.obj', mesh_resolution=1024):
    model.eval()
    # model needs ( rgb, depth, nomral ), implement img to depth and normal.
    depth = get_depth_map(img, corners)
    depth = torch.from_numpy(depth.transpose(2, 0, 1)[None]).to(args.device)
    normal = get_normal_map(img, corners)
    normal = torch.from_numpy(normal.transpose(2, 0, 1)[None]).to(args.device)
    dt = model(torch.from_numpy(img.transpose(2, 0, 1)[None]).to(args.device), depth, normal)
    # dt[] only got trivialWalls key

    # if args.post_processing != 'original':
    #     dt['processed_xyz'] = post_process(tensor2np(dt['depth']), type_name=args.post_processing)

    #output_xyz = dt['processed_xyz'][0] if 'processed_xyz' in dt else depth2xyz(tensor2np(dt['depth'][0]))

    #json_data = save_pred_json(output_xyz, tensor2np(dt['ratio'][0])[0],
    #                           save_path=os.path.join(args.output_dir, f"{name}_pred.json"))
    
    
    floor_pts = np.round(corners[len(corners)//2:] * [256,128]).astype(np.uint16)
    min_value = np.min(floor_pts[:,0])
    loop_cnt = 0
    while floor_pts[0,0] != min_value:
        assert loop_cnt < 50, 'infinite looping'
        loop_cnt+=1
        floor_pts = np.roll(floor_pts, shift = -1, axis = 0)
    wall_tw = np.zeros(256)
    max_tw = 0
    wall_id = 0
    # table是用來記所有牆壁的TW值包括遮擋牆，而visible的則是用來visualize用的，所以不需要記遮擋牆
    table = []
    visible_table = []
    for i in range(len(floor_pts)-1):
        # occluded wall
        if floor_pts[i,0] >= floor_pts[i+1,0]:
            # 遮擋的一率為零
            table.append(0)
        else:
            tw = cal_tw(floor_pts[i,0],floor_pts[i+1,0],dt['trivialWalls'][0].cpu().numpy())
            table.append(tw)
            visible_table.append(tw)
       
    last_tw = cal_tw(floor_pts[0,0],floor_pts[-1,0],dt['trivialWalls'][0].cpu().numpy(),last_wall=True)
    table.append(last_tw)
    visible_table.append(last_tw)
    table = np.array(table)
    visible_table = np.array(visible_table)
    table = np.clip(table, 0, 1)
    visible_table = np.clip(visible_table, 0, 1)
    # 存TW 以牆壁為單位
    bin_table = [1 if a >= 0.5 else 0 for a in table]
    with open(os.path.join(args.output_dir,f'{name}_TW.txt'), 'w') as file:
        for num in bin_table:
            file.write(f'{num}\n')  # Write each number followed by a newline
    # 存後處理過後的1d vector 回預測結果 (ie. 把平均值大於0.5的標記成1, 反之)
    for i in range(len(visible_table)):
        if i != len(visible_table)-1:
            wall_tw[floor_pts[i,0]:floor_pts[i+1,0]] = visible_table[i]
        else:
            wall_tw[:floor_pts[0,0]] = visible_table[i]
            wall_tw[floor_pts[i,0]:] = visible_table[i]
    wall_tw_tensor = torch.from_numpy(wall_tw)



    dt['trivialWalls'][0] = wall_tw_tensor
    save_name = name+"_pred.png"
    visualize_2d(img, corners, dt,
                show_depth=show_depth,
                show_floorplan=show_floorplan,
                show=show,
                save_path=os.path.join(args.output_dir, save_name))

if __name__ == '__main__':
    logger = get_logger()
    args = parse_option()
    config = get_config(args)

    if ('cuda' in args.device or 'cuda' in config.TRAIN.DEVICE) and not torch.cuda.is_available():
        logger.info(f'The {args.device} is not available, will use cpu ...')
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()

    model, _, _, _ = build_model(config, logger)
    os.makedirs(args.output_dir, exist_ok=True)
    img_paths = sorted(glob.glob(args.data_glob + '/*.jpg') + glob.glob(args.data_glob + '/*.png'))
    corners_paths = sorted(glob.glob(args.data_glob+'/*.txt'))
    inference()

    # dataset = MP3DDataset(root_dir='./src/dataset/mp3d', mode='test', split_list=[
    #     ['7y3sRwLe3Va', '155fac2d50764bf09feb6c8f33e8fb76'],
    #     ['e9zR4mvMWw7', 'c904c55a5d0e420bbd6e4e030b9fe5b4'],
    # ])
    # dataset = ZindDataset(root_dir='./src/dataset/zind', mode='test', split_list=[
    #     '1169_pano_21',
    #     '0583_pano_59',
    # ], vp_align=True)
    # inference_dataset(dataset)
