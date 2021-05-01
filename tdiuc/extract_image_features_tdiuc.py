import argparse, os
import h5py
import numpy as np
from scipy.misc import imread, imresize

import torch
import torchvision
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/media/qzhb/DATA1/yi/dorren/tdiuc/')
parser.add_argument('--dataset', default='tdiuc', type=str, choices=['tdiuc', 'VQAv2'])
parser.add_argument('--max_images', default=None, type=int)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet152')
parser.add_argument('--model_stage', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()
        return feats


def path2iid(path):
    return int(path.split('/')[-1].split('.')[0].split('_')[-1])


def main(args):
    args.output_h5_file = args.path + "/all_{DATASET}_features.h5"
    p1 = f'{args.path}/Images/train2014'
    input_paths = [os.path.join(p1, a) for a in os.listdir(p1)]

    p1 = f'{args.path}/Images/val2014'
    input_paths.extend([os.path.join(p1, a) for a in os.listdir(p1)])

    model = build_model(args)
    img_size = (args.image_height, args.image_width)
    with h5py.File(args.output_h5_file, 'w') as f:
        feat_dset = None
        i0 = 0
        cur_batch = []
        iid = []
        for i, path in enumerate(input_paths):
            iid.append(path2iid(path))
            img = imread(path, mode='RGB')
            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]  # (224,224,3)  -->  (1,3,224,224)
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset('image_features', (N, H * W, C),
                                                 dtype=np.float32)
                    img_dset = f.create_dataset('image', (N, 1, 3, args.image_width, args.image_height),dtype=np.float32)
                    iid_dset = f.create_dataset('iids', (N,),
                                                dtype=np.int64)

                i1 = i0 + len(cur_batch)
                feats_r = feats.reshape(-1, 2048, 49)
                feat_dset[i0:i1] = np.transpose(feats_r, (0, 2, 1))
                img_dset[i0:i1] = np.array(cur_batch)
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)))
                cur_batch = []

        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model)
            feats_r = feats.reshape(-1, 2048, 49)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = np.transpose(feats_r, (0, 2, 1))
            img_dset[i0:i1] = np.array(cur_batch)
            print('Processed %d / %d images' % (i1, len(input_paths)))
        iid_dset[:len(iid)] = np.array(iid, dtype=np.int64)

        feat_file = h5py.File(args.output_h5_file, 'r')

        iid_list = feat_file['iids'][:]

        iid2idx = {str(iid): idx for idx, iid in enumerate(iid_list)}
        idx2iid = {idx: str(iid) for idx, iid in enumerate(iid_list)}

        lut = dict()
        lut['image_id_to_ix'] = iid2idx
        lut['image_ix_to_id'] = idx2iid

        json.dump(lut, open(f'{args.path}/map_{args.dataset}_features.json', 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
