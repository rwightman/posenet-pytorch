import urllib.request
import posixpath
import json
import os

from posenet import MOBILENET_V1_CHECKPOINTS

GOOGLE_CLOUD_STORAGE_DIR = 'https://storage.googleapis.com/tfjs-models/weights/posenet/'


def download_file(checkpoint, filename, base_dir):
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    urllib.request.urlretrieve(url, os.path.join(base_dir, checkpoint, filename))


def download(checkpoint, base_dir='./weights/'):
    save_dir = os.path.join(base_dir, checkpoint)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download_file(checkpoint, 'manifest.json', base_dir)

    f = open(os.path.join(save_dir, 'manifest.json'), 'r')
    json_dict = json.load(f)

    for x in json_dict:
        filename = json_dict[x]['filename']
        print('Downloading', filename)
        download_file(checkpoint, filename, base_dir)


def main():
    checkpoint = MOBILENET_V1_CHECKPOINTS[101]
    download(checkpoint)


if __name__ == "__main__":
    main()
