import os
import numpy as np
from utils.conversion import xyz2pixel
import json
from tqdm import tqdm

def get_cor(anno):
    try:
        corner_xz = np.array(anno['layout_raw']['vertices'])
        corner_xz[..., 0] = -corner_xz[..., 0]
        corner_xyz = np.insert(corner_xz, 1, anno['camera_height'], axis=1)
        ratio = np.array([(anno['ceiling_height'] - anno['camera_height']) / anno['camera_height']], dtype=np.float32)
        ceil = corner_xyz * [1, -1 * ratio.item(), 1]
        infer_corners = xyz2pixel(np.concatenate([ceil, corner_xyz]))
        return infer_corners
    except KeyError as e:
        print(f"KeyError: {e} in annotation data: {anno}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

if __name__ == '__main__':
    data_dir = "/media/user/WD_BLACK/noah/zind/datasets"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The data directory {data_dir} does not exist.")
    
    indices = os.listdir(data_dir)
    for index in tqdm(indices):
        try:
            with open(os.path.join(data_dir, index, "zind_data.json")) as file:
                anno = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {os.path.join(data_dir, index, 'zind_data.json')}")
            continue
        except json.JSONDecodeError:
            print(f"JSON decode error in file: {os.path.join(data_dir, index, 'zind_data.json')}")
            continue

        try:
            anno = anno['merger']
        except KeyError:
            print(f"'merger' key not found in annotation data for index: {index}")
            continue

        for floor in anno:
            for complete_room in anno[floor]:
                for partial_room in anno[floor][complete_room]:
                    for pano in anno[floor][complete_room][partial_room]:
                        try:
                            now = anno[floor][complete_room][partial_room][pano]
                            infer_cor = get_cor(now)
                            if infer_cor is None:
                                print(f"Skipping {pano} due to error in get_cor function.")
                                continue
                            img_path = now['image_path']
                            name = img_path.split('/')[1]
                            output_path = os.path.join(data_dir, index, 'panos_aligned', name + '.txt')
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, 'w') as data:
                                for cor in infer_cor:
                                    data.write(' '.join(map(str, cor)) + '\n')
                        except KeyError as e:
                            print(f"KeyError: {e} while processing pano: {pano}")
                        except Exception as e:
                            print(f"Unexpected error: {e} while processing pano: {pano}")
