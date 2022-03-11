
# from PIL import Image
# import io
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


def video_decorder(jpg_byte_list, clip_sampler, transform):

    frame_indices = clip_sampler(list(range(len(jpg_byte_list))))

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


def fixed_clip_sampler(frame_indices):
    return frame_indices[::16]


def wds_dataset(
    shards_url,
    clip_sampler,
    transform=None,
    shuffle_buffer_size=10,
):
    dataset = (
        wds.WebDataset(shards_url)
        .shuffle(shuffle_buffer_size)
        .decode(
            wds.handle_extension('video.pickle', pickle_decoder),
            wds.handle_extension('stats.pickle', pickle_decoder),
        )
        .to_tuple('video.pickle', 'stats.pickle')
        .map_tuple(
            partial(
                video_decorder,
                clip_sampler=clip_sampler,
                transform=transform,
            ),
            get_label,
        )
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
        clip_sampler=fixed_clip_sampler,
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
            if args.decorder == 'dali':
                video = batch[0]['video'].to(device)  # torch.Tensor of BCTHW
                label = batch[0]['label'].to(device)  # torch.Tensor of [B, 1]
                label = label.squeeze()  # [B]
            else:
                video = batch['video'].to(device)
                label = batch['label'].to(device)
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
                        default='./shards/UCF101-{00000..00004}.tar',
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

    args = parser.parse_args()
    print(args)
    main(args)
