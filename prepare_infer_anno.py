import os
import numpy as np
from utils.conversion import xyz2pixel
import json
from tqdm import tqdm

def get_cor(anno):
    corner_xz = np.array(anno['layout_raw']['vertices'])
    corner_xz[..., 0] = -corner_xz[..., 0]
    corner_xyz = np.insert(corner_xz, 1, anno['camera_height'], axis=1)
    ratio = np.array([(anno['ceiling_height'] - anno['camera_height']) / anno['camera_height']], dtype=np.float32)
    ceil = corner_xyz*[1,-1*ratio.item(),1]
    infer_corners = xyz2pixel(np.concatenate([ceil,corner_xyz]))
    return infer_corners

if __name__ == '__main__':
    data_dir = "/media/user/WD_BLACK/noah/zind/datasets"
    indices = os.listdir(data_dir)
    for index in tqdm(indices):
        with open(os.path.join(data_dir,index,"zind_data.json")) as file:
            anno = json.load(file)
        anno = anno['merger']
        for floor in anno:
            for complete_room in anno[floor]:
                for partial_room in anno[floor][complete_room]:
                    for pano in anno[floor][complete_room][partial_room]:
                        now = anno[floor][complete_room][partial_room][pano]
                        infer_cor = get_cor(now)
                        img_path = now['image_path']
                        name = img_path.split('/')[1]
                        with open(os.path.join(data_dir,index,'panos',name+'.txt'),'w') as data:
                            for cor in infer_cor:
                                data.write(' '.join(map(str, cor)) + '\n') 
