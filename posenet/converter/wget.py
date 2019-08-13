import requests
import json
import posixpath
import os

from posenet import MOBILENET_V1_CHECKPOINTS

GOOGLE_CLOUD_STORAGE_DIR = 'https://storage.googleapis.com/tfjs-models/weights/posenet/'


def download_json(checkpoint, filename, base_dir):

    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url)
    data = json.loads(response.content)

    with open(os.path.join(base_dir, checkpoint, filename), 'w') as outfile:
        json.dump(data, outfile)

def download_file(checkpoint, filename, base_dir):

    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url)
    f = open(os.path.join(base_dir, checkpoint, filename), 'wb')
    f.write(response.content)
    f.close()

def download(checkpoint, base_dir='./weights/'):
    save_dir = os.path.join(base_dir, checkpoint)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download_json(checkpoint, 'manifest.json', base_dir)

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
