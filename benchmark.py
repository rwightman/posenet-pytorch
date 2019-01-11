import torch
import time
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()


def main():

    with torch.no_grad():
        model = posenet.load_model(args.model)
        model = model.cuda()
        output_stride = model.output_stride
        height = width = 513
        num_images = args.num_images

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        if len(filenames) > num_images:
            filenames = filenames[:num_images]

        images = {f: posenet.read_imgfile(f, width, height)[0] for f in filenames}

        start = time.time()
        for i in range(num_images):
            input_image = torch.Tensor(images[filenames[i % len(filenames)]]).cuda()

            results = model(input_image)
            heatmaps, offsets, displacement_fwd, displacement_bwd = results
            output = posenet.decode_multiple_poses(
                heatmaps.squeeze(0),
                offsets.squeeze(0),
                displacement_fwd.squeeze(0),
                displacement_bwd.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        print('Average FPS:', num_images / (time.time() - start))


if __name__ == "__main__":
    main()
