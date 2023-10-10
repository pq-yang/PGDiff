import os
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw

def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 12
    # test very large mask ratio
    min_width = 80
    max_width = 120
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        append_mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img 

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            draw2 = ImageDraw.Draw(append_mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            draw2.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)
                draw2.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask, append_mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', type=str, help="Input folder for color statistics calculation.")
    parser.add_argument('-o', '--out_path', type=str, help="Output folder for color statistics calculation.")
    args = parser.parse_args()

    img_folder = args.in_path
    out_folder = args.out_path
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(f'{out_folder}/mask', exist_ok=True)
    os.makedirs(f'{out_folder}/append_mask', exist_ok=True)

    img_list = os.listdir(img_folder)
    for img_name in img_list:
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path)
        mask, append_mask = brush_stroke_mask(img)
        mask.save(f'{out_folder}/mask/{img_name}')
        append_mask.save(f'{out_folder}/append_mask/{img_name}')