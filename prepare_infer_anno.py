import os
import numpy as np
from utils.conversion import xyz2pixel
import json
from tqdm import tqdm
import cv2

def get_cor(anno):
    try:
        corner_xz = np.array(anno['layout_raw']['vertices'])
        corner_xz[..., 0] = -corner_xz[..., 0]
        corner_xyz = np.insert(corner_xz, 1, anno['camera_height'], axis=1)
        ratio = (anno['ceiling_height'] - anno['camera_height']) / anno['camera_height']
        ceil = corner_xyz * [1, -ratio, 1]
        infer_corners = xyz2pixel(np.concatenate([ceil, corner_xyz]))
        return infer_corners
    except KeyError as e:
        print(f"KeyError: {e} in annotation data: {anno}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def main():
    data_dir = "/media/user/WD_BLACK/noah/zind/datasets"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The data directory {data_dir} does not exist.")

    with open('./partition.txt', 'r') as file:
        indices = [line.strip() for line in file]

    count = 0
    for index in tqdm(indices):
        id = index.split('/')
        cur_pano = '_'.join(id[9].replace('.png', '').split('_')[5:7])
        
        try:
            with open(os.path.join(data_dir, id[7], "zind_data.json")) as file:
                anno = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e} in file: {os.path.join(data_dir, id[7], 'zind_data.json')}")
            continue

        try:
            merger = anno['merger']
        except KeyError:
            print(f"'merger' key not found in annotation data for index: {id[7]}")
            continue
        
        for floor, complete_rooms in merger.items():
            for complete_room, partial_rooms in complete_rooms.items():
                for partial_room, panos in partial_rooms.items():
                    for pano, pano_data in panos.items():
                        if pano.strip() == cur_pano:
                            infer_cor = get_cor(pano_data)
                            if infer_cor is None:
                                print(f"Skipping {pano} due to error in get_cor function.")
                                continue
                            
                            img_path = pano_data['image_path']
                            img_name = img_path.split('/')[1].replace('.jpg', '')
                            img = cv2.imread(os.path.join(data_dir, id[7], img_path))
                            img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)

                            output_dir = os.path.join(data_dir, id[7], 'panos_test')
                            os.makedirs(output_dir, exist_ok=True)

                            cv2.imwrite(os.path.join(output_dir, f'{img_name}.jpg'), img)
                            output_txt_path = os.path.join(output_dir, f'{img_name}.txt')
                            with open(output_txt_path, 'w') as data_file:
                                for cor in infer_cor:
                                    data_file.write(' '.join(map(str, cor)) + '\n')
                            
                            count += 1
    
    print('Finished:', count)

if __name__ == '__main__':
    main()
