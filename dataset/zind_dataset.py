""" 
@Date: 2021/09/22
@description:
"""
import os
import json
import math
import numpy as np
import torch
from dataset.communal.read import read_image, read_label, read_zind
from dataset.communal.base_dataset import BaseDataset
from utils.logger import get_logger
from preprocessing.filter import filter_center, filter_boundary, filter_self_intersection
from utils.boundary import calc_rotation, boundary_type, visibility_corners
from visualization.boundary import draw_walls
from utils.conversion import uv2xyz, depth2xyz
from visualization.grad import convert_img




class ZindDataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None,
                 is_simple=True, is_ceiling_flat=False, vp_align=False):
        # if keys is None:
        #     keys = ['image', 'depth', 'ratio', 'id', 'corners', 'corner_heat_map', 'object']
        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)
        if logger is None:
            logger = get_logger()
        self.root_dir = root_dir
        self.vp_align = vp_align

        data_dir = os.path.join(root_dir)
        img_dir = os.path.join(root_dir, 'image')
        
        # 保存那些boundary_type error
        error = []

        pano_list = read_zind(partition_path=os.path.join(data_dir, f"zind_partition.json"),
                              simplicity_path=os.path.join(data_dir, f"room_shape_simplicity_labels.json"),
                              data_dir=data_dir, mode=mode, is_simple=is_simple, is_ceiling_flat=is_ceiling_flat)

        if for_test_index is not None:
            pano_list = pano_list[:for_test_index]
        if split_list:
            pano_list = [pano for pano in pano_list if pano['id'] in split_list]
        self.data = []
        invalid_num = 0
        for pano in pano_list:
            if not os.path.exists(pano['img_path']):
                logger.warning(f"{pano['img_path']} not exists")
                invalid_num += 1
                continue

            if not filter_center(pano['corners']):
                logger.warning(f"{pano['id']} camera center not in layout")
                invalid_num += 1
                continue

            if self.max_wall_num >= 10:
                if len(pano['corners']) < self.max_wall_num:
                    invalid_num += 1
                    continue
            elif self.max_wall_num != 0 and len(pano['corners']) != self.max_wall_num:
                invalid_num += 1
                continue

            if not filter_boundary(pano['corners']):
                logger.warning(f"{pano['id']} boundary cross")
                invalid_num += 1
                continue

            if not filter_self_intersection(pano['corners']):
                logger.warning(f"{pano['id']} self_intersection")
                invalid_num += 1
                error.append(pano['img_path']+'/'+pano['id'])
                continue

            if 'trivialWalls' not in pano:
                logger.warning(f"{pano['id']} trivialWalls key error")
                invalid_num += 1
                continue

            if pano['trivialWalls'].any() == None:
                logger.warning(f"{pano['id']} trivialWalls key == None")
                invalid_num += 1
                continue

            
            if boundary_type(pano['uv_corners_list'][0]) is None or boundary_type(pano['uv_corners_list'][1]) is None:
                invalid_num +=1
                logger.warning(f"{pano['id']} boundary error!!")
                continue

            self.data.append(pano)

        with open('./error.txt','w') as file:
            for pano in error:
                file.write(pano+'\n')
        logger.info(
            f"Build dataset mode: {self.mode} max_wall_num: {self.max_wall_num} valid: {len(self.data)} invalid: {invalid_num} error: {len(error)}")

    def __getitem__(self, idx):
        pano = self.data[idx]
        rgb_path = pano['img_path']
        label = pano
        image = read_image(rgb_path, self.shape) # (512, 1024, 3)
        # depth map
        visible_corners = visibility_corners(label['corners'])
        depth = self.get_depth(visible_corners, length=image.shape[1], visible=True) 
        depth_img = np.expand_dims(depth,axis=0) # [1, pathc_num]
        depth_img = np.repeat(depth_img,image.shape[0],axis=0) # [patch_num//2, patch_num]
        depth_img = np.expand_dims(depth_img,axis=-1)
        depth_img = np.repeat(depth_img,3,axis=-1)
        depth_img = draw_walls(depth_img,label['uv_corners_list'],ch_num=1)
        # nomral map
        depth = torch.tensor(depth)
        xz = depth2xyz(depth)[:,::2]
        direction = torch.roll(xz, -1, dims=0) - xz  # direct[i] = xz[i+1] - xz[i]
        direction = direction / direction.norm(p=2, dim=-1)[..., None]
        angle = torch.atan2(direction[..., 1], direction[..., 0])
        normal_img = convert_img(angle, image.shape[0], cmap='HSV')
        normal_img = draw_walls(normal_img,label['uv_corners_list'])

        if self.vp_align:
            #  Equivalent to vanishing point alignment step
            rotation = calc_rotation(corners=label['corners'])
            shift = math.modf(rotation / (2 * np.pi) + 1)[0]
            image = np.roll(image, round(shift * self.shape[1]), axis=1)
            depth_img = np.roll(depth_img, round(shift * self.shape[1]), axis=1)
            normal_img = np.roll(normal_img, round(shift * self.shape[1]), axis=1)

            label['trivialWalls'] = np.roll(label['trivialWalls'], round(shift * 256))
            label['corners'][:, 0] = np.modf(label['corners'][:, 0] + shift)[0]
            # # cei
            # label['uv_corners_list'][0][:, 0] = label['corners'][:, 0]
            # # flo
            # label['uv_corners_list'][1] = label['corners']

        output = self.process_data(label, image, depth_img, normal_img, self.patch_num)
        return output


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    from tqdm import tqdm
    from visualization.boundary import draw_boundaries, draw_object
    from visualization.floorplan import draw_floorplan
    from utils.boundary import depth2boundaries, calc_rotation
    from utils.conversion import uv2xyz
    from models.other.init_env import init_env

    init_env(123)

    modes = ['val']
    for i in range(1):
        for mode in modes:
            print(mode)
            mp3d_dataset = ZindDataset(root_dir='../src/dataset/zind', mode=mode, aug={
                'STRETCH': False,
                'ROTATE': False,
                'FLIP': False,
                'GAMMA': False
            })
            # continue
            # save_dir = f'../src/dataset/zind/visualization/{mode}'
            # if not os.path.isdir(save_dir):
            #     os.makedirs(save_dir)

            bar = tqdm(mp3d_dataset, ncols=100)
            for data in bar:
                # if data['id'] != '1079_pano_18':
                #     continue
                bar.set_description(f"Processing {data['id']}")
                boundary_list = depth2boundaries(data['ratio'], data['depth'], step=None)

                pano_img = draw_boundaries(data['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
                # Image.fromarray((pano_img * 255).astype(np.uint8)).save(
                #     os.path.join(save_dir, f"{data['id']}_boundary.png"))
                # draw_object(pano_img, heat_maps=data['object_heat_map'], depth=data['depth'],
                #             size=data['object_size'], show=True)
                # pass
                #
                floorplan = draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=True,
                                           marker_color=None, center_color=0.2)
                # Image.fromarray((floorplan.squeeze() * 255).astype(np.uint8)).save(
                #     os.path.join(save_dir, f"{data['id']}_floorplan.png"))
