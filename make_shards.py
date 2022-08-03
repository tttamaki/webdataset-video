from pathlib import Path
from tqdm import tqdm
import argparse
import os
import webdataset as wds
import av
import random
import json


def bytes2kmg(b: int) -> str:
    GB = 1024 * 1024 * 1024
    MB = 1024 * 1024
    kB = 1024
    if b > GB:
        return str(b // GB) + 'GB'
    elif b > MB:
        return str(b // MB) + 'MB'
    elif b > kB:
        return str(b // kB) + 'kB'
    else:
        return str(b)


def make_shards(args):

    video_file_paths = list(
        Path(args.dataset_path).glob(f'**/*.{args.video_ext}')
    )
    if args.shuffle:
        random.shuffle(video_file_paths)

    # https://github.com/pytorch/vision/blob/a8bde78130fd8c956780d85693d0f51912013732/torchvision/datasets/folder.py#L36
    class_list = sorted(
        entry.name for entry in os.scandir(args.dataset_path)
        if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    shard_dir_path = Path(args.shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / f'{args.shard_prefix}-%05d.tar')

    with wds.ShardWriter(
        pattern=shard_filename,
        maxsize=args.max_size,
        maxcount=args.max_count
    ) as sink, tqdm(
        video_file_paths,
        total=len(video_file_paths),
    ) as pbar_path:

        for video_file_path in pbar_path:

            pbar_path.set_postfix_str(f"tar size: {bytes2kmg(sink.size)}")

            video_stream_id = 0  # default

            with open(str(video_file_path), "rb") as f:
                movie_binary = f.read(-1)

            container = av.open(str(video_file_path))
            stream = container.streams.video[video_stream_id]

            if stream.frames > 0:
                n_frames = stream.frames
            else:
                # stream.frames is not available for some codecs
                n_frames = int(float(container.duration)
                               / av.time_base * stream.base_rate)

            category_name = video_file_path.parent.name
            label = class_to_idx[category_name]
            key_str = category_name + '/' + video_file_path.stem

            video_stats_json = json.dumps({
                '__key__': key_str,
                'video_id': video_file_path.stem,
                'filename': video_file_path.name,
                'suffix': video_file_path.suffix[1:],  # remove "." from ".avi"
                'category': category_name,
                'label': label,
                'width': stream.codec_context.width,
                'height': stream.codec_context.height,
                'fps': float(stream.base_rate),
                'n_frames': n_frames,
                'duraion': float(container.duration) / av.time_base,
            })

            sample_dic = {
                '__key__': key_str,
                'video.bin': movie_binary,
                'stats.json': video_stats_json,
            }
            sink.write(sample_dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_path', action='store',
                        default='/mnt/NAS-TVS872XT/dataset/UCF101/video/',
                        help='Path to the dataset dir with category subdirs.')
    parser.add_argument('-e', '--video_ext',
                        choices=['mp4', 'avi'],
                        default='avi',
                        help='Extension of video files. mp4 or avi. '
                        'default to avi.')
    parser.add_argument('-s', '--shard_path', action='store',
                        default='./shards_video/',
                        help='Path to the dir to store shard tar files.')
    parser.add_argument('-p', '--shard_prefix', action='store',
                        default='UCF101',
                        help='Prefix of shard tar files.')
    parser.add_argument('-q', '--quality', type=int, default=80,
                        help='Qualify factor of JPEG file. '
                        'default to 80.')
    parser.add_argument('--max_size', type=int, default=1024 * 1024 * 1024,
                        help='Max size of each shard tar file. '
                        'default to 1GB.')
    parser.add_argument('--max_count', type=int, default=10000,
                        help='Max number of entries in each shard tar file. '
                        'default to 10,000.')

    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='use shuffle')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='do not use shuffle')
    parser.set_defaults(shuffle=True)

    args = parser.parse_args()

    make_shards(args)
