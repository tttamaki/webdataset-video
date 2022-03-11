from io import BytesIO
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import cv2
import os
import webdataset as wds


def make_shards(args):

    video_file_paths = list(
        Path(args.dataset_path).glob(f'**/*.{args.video_ext}')
    )

    # https://github.com/pytorch/vision/blob/a8bde78130fd8c956780d85693d0f51912013732/torchvision/datasets/folder.py#L36
    class_list = sorted(
        entry.name for entry in os.scandir(args.dataset_path)
        if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    shard_dir_path = Path(args.shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    pattern = str(shard_dir_path / f'{args.shard_prefix}-%05d.tar')

    with wds.ShardWriter(
        pattern,
        maxsize=args.max_size,
        maxcount=args.max_count
    ) as sink, tqdm(
        video_file_paths,
        total=len(video_file_paths),
    ) as path_pbar:

        for video_file_path in path_pbar:

            jpg_byte_list = []
            cap = cv2.VideoCapture(str(video_file_path))
            if not cap.isOpened():
                continue
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                with BytesIO() as buffer:
                    img.save(buffer,
                             format='JPEG',
                             quality=args.quality)
                    jpg_byte_list.append(buffer.getvalue())

            category_name = video_file_path.parent.name
            label = class_to_idx[category_name]
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            video_stats_dic = {
                'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': fps,
                'frame_count': frame_count,
                'len': len(jpg_byte_list),  # == frame_count
                'duraion': frame_count * fps,
                'category': category_name,
                'label': label,
            }

            sink.write({
                '__key__': video_file_path.stem,
                'video.pickle': jpg_byte_list,
                'label.txt': str(label),
                'stats.pickle': video_stats_dic,
            })


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
                        default='./shards/',
                        help='Path to the dir to store shard tar files.')
    parser.add_argument('-p', '--shard_prefix', action='store',
                        default='UCF101',
                        help='Prefix of shard tar files.')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='Qualify factor of JPEG file. '
                        'default to 95.')
    parser.add_argument('--max_size', type=int, default=1e10,
                        help='Max size of each shard tar file. '
                        'default to 10GB.')
    parser.add_argument('--max_count', type=int, default=100000,
                        help='Max number of entries in each shard tar file. '
                        'default to 100,000.')
    args = parser.parse_args()

    make_shards(args)
