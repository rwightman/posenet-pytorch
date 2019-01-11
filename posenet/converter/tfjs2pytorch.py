import json
import struct
import cv2
import numpy as np
import os
import tempfile
import torch

from posenet import MobileNetV1, MOBILENET_V1_CHECKPOINTS


BASE_DIR = os.path.join(tempfile.gettempdir(), '_posenet_weights')


def to_torch_name(tf_name):
    tf_name = tf_name.lower()
    tf_split = tf_name.split('/')
    tf_layer_split = tf_split[1].split('_')
    tf_variable_type = tf_split[2]
    if tf_variable_type == 'weights' or tf_variable_type == 'depthwise_weights':
        variable_postfix = '.weight'
    elif tf_variable_type == 'biases':
        variable_postfix = '.bias'
    else:
        variable_postfix = ''

    if tf_layer_split[0] == 'conv2d':
        torch_name = 'features.conv' + tf_layer_split[1]
        if len(tf_layer_split) > 2:
            torch_name += '.' + tf_layer_split[2]
        else:
            torch_name += '.conv'
        torch_name += variable_postfix
    else:
        if tf_layer_split[0] in ['offset', 'displacement', 'heatmap'] and tf_layer_split[-1] == '2':
            torch_name = '_'.join(tf_layer_split[:-1])
            torch_name += variable_postfix
        else:
            torch_name = ''

    return torch_name


def load_variables(chkpoint, base_dir=BASE_DIR):
    manifest_path = os.path.join(base_dir, chkpoint, "manifest.json")
    if not os.path.exists(manifest_path):
        print('Weights for checkpoint %s are not downloaded. Downloading to %s ...' % (chkpoint, base_dir))
        from posenet.converter.wget import download
        download(chkpoint, base_dir)
        assert os.path.exists(manifest_path)

    manifest = open(manifest_path)
    variables = json.load(manifest)
    manifest.close()

    state_dict = {}
    for x in variables:
        torch_name = to_torch_name(x)
        if not torch_name:
            continue
        filename = variables[x]["filename"]
        byte = open(os.path.join(base_dir, chkpoint, filename), 'rb').read()
        fmt = str(int(len(byte) / struct.calcsize('f'))) + 'f'
        d = struct.unpack(fmt, byte)
        d = np.array(d, dtype=np.float32)
        shape = variables[x]["shape"]
        if len(shape) == 4:
            tpt = (2, 3, 0, 1) if 'depthwise' in filename else (3, 2, 0, 1)
            d = np.reshape(d, shape).transpose(tpt)
        state_dict[torch_name] = torch.Tensor(d)

    return state_dict


def read_imgfile(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    img = img.transpose((2, 0, 1))
    return img


def convert(model_id, model_dir, output_stride=16, image_size=513, check=True):
    checkpoint_name = MOBILENET_V1_CHECKPOINTS[model_id]
    width = image_size
    height = image_size

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    state_dict = load_variables(checkpoint_name)
    m = MobileNetV1(model_id, output_stride=output_stride)
    m.load_state_dict(state_dict)
    checkpoint_path = os.path.join(model_dir, checkpoint_name) + '.pth'
    torch.save(m.state_dict(), checkpoint_path)

    if check and os.path.exists("./tennis_in_crowd.jpg"):
        # Result
        input_image = read_imgfile("./tennis_in_crowd.jpg", width, height)
        input_image = np.array(input_image, dtype=np.float32)
        input_image = input_image.reshape(1, 3, height, width)
        input_image = torch.Tensor(input_image)

        heatmaps_result, offset_result, displacement_fwd_result, displacement_bwd_result = m(input_image)

        print("Heatmaps")
        print(heatmaps_result.shape)
        print(heatmaps_result[:, 0:1, 0:1])
        print(torch.mean(heatmaps_result))

