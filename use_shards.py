
from average_meter import AverageMeter
from functools import partial
from torch import as_tensor
from torch.utils.data import DataLoader
from torchvision.io import decode_jpeg
from tqdm import tqdm
import argparse
import pickle
import time
import torch
import webdataset as wds
import numpy as np


def video_decorder(jpg_byte_list, clip_sampler, transform, n_frames):

    frame_indices = clip_sampler(list(range(len(jpg_byte_list))), n_frames)

    clip = [decode_jpeg(as_tensor(list(jpg_byte_list[i]),
                                  dtype=torch.uint8))
            for i in frame_indices]
    clip = torch.stack(clip, 0)  # TCHW
    # clip = transform(clip)

    return clip


def get_label(value):
    return value['label']


def pickle_decoder(value):
    return pickle.loads(value)


def identity(value):
    return value


def uniform_clip_sampler(frame_indices, n_frames):
    # return frame_indices[::16]
    return np.linspace(0, len(frame_indices) - 1, num=n_frames).astype(int)


def wds_dataset(
    shards_url,
    clip_sampler,
    n_frames=8,
    transform=None,
    shuffle_buffer_size=10,
):
    dataset = wds.WebDataset(shards_url)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.decode(
        wds.handle_extension('video.pickle', pickle_decoder),
        # wds.handle_extension('timestamp.pickle', pickle_decoder),
        wds.handle_extension('stats.pickle', pickle_decoder),
    )
    dataset = dataset.to_tuple(
        'video.pickle',
        'stats.pickle',
    )
    dataset = dataset.map_tuple(
        # 'video.pickle'
        partial(
            video_decorder,
            clip_sampler=clip_sampler,
            transform=transform,
            n_frames=n_frames,
        ),
        # 'stats.pickle'
        get_label,
    )
    dataset = dataset.with_length(13000)  # dummy

    return dataset


def main(args):

    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    et_data_load = AverageMeter()
    et_to_device = AverageMeter()

    dataset = wds_dataset(
        shards_url=args.shards_url,
        clip_sampler=uniform_clip_sampler,
        n_frames=args.n_frames,
    )
    sample_loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    st = time.time()

    with tqdm(enumerate(sample_loader),
              total=len(sample_loader),
              leave=True,
              smoothing=0,
              ) as pbar_batch:

        for i, batch in pbar_batch:

            et_data_load.update(time.time() - st)

            st = time.time()
            video = batch[0].to(device)
            label = batch[1].to(device)
            et_to_device.update(time.time() - st)

            print('batch index', i)
            print('video.shape', video.shape)
            print('label.shape', label.shape)

            pbar_batch.set_postfix_str(
                'data load {:6.04f} '
                'to_device {:6.04f} '
                .format(
                    et_data_load.avg,
                    et_to_device.avg,
                ))

            st = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--shards_url', action='store',
                        default='./shards/UCF101-{00000..00002}.tar',
                        # default='./shards/UCF101-00000.tar',
                        help='Path to the dir to store shard tar files.')
    parser.add_argument('--shuffle', type=int, default=10,
                        help='shuffle buffer size. default 10')

    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size. default 8')
    parser.add_argument('-w', '--num_workers', type=int, default=12,
                        help='number of dataloader workders. default 12')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='GPU No. to be used for model. default 0')

    parser.add_argument('-f', '--n_frames', type=int, default=8,
                        help='number of frames in a batch. default 8')

    args = parser.parse_args()
    print(args)
    main(args)
