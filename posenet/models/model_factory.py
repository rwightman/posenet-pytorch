import torch
import os


from posenet.models.mobilenet_v1 import MobileNetV1, MOBILENET_V1_CHECKPOINTS

MODEL_DIR = './_models'
DEBUG_OUTPUT = False


def load_model(model_id, output_stride=16, model_dir=MODEL_DIR):
    model_path = os.path.join(model_dir, MOBILENET_V1_CHECKPOINTS[model_id] + '.pth')
    if not os.path.exists(model_path):
        print('Cannot find models file %s, converting from tfjs...' % model_path)
        from posenet.converter.tfjs2pytorch import convert
        convert(model_id, model_dir, check=False)
        assert os.path.exists(model_path)

    model = MobileNetV1(model_id, output_stride=output_stride)
    load_dict = torch.load(model_path)
    model.load_state_dict(load_dict)

    return model
