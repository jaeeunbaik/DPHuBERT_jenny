import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Union, Iterator, Dict, Optional
from torch import Tensor
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
import cv2
import torchvision
import random
import os
import h5py
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class BucketizeBatchSampler(BatchSampler):
    """Bucketized BatchSampler for sequential video data with different lengths to reduce number of paddings.
    Args:
        lengths (List[int]): The lengths (number of frames) of the samples in the dataset.
        num_buckets (int): The number of buckets to split the data samples.
        frame_rate (int): Frame rate of the videos (default: 25).
        min_len (int, optional): The minimum sample lengths to keep.
            (Default: 0)
        max_len (int or None, optional): The maximum sample lengths to keep. Inferred if not provided.
            (Default ``None``)
        max_token_count (float or None, optional): The max number of seconds in one mini-batch.
            (Default: ``None``)
        batch_size (int or None, optional): The number of samples in one mini-batch.
            (Default: ``None``)
        shuffle (bool, optional): Whether to shuffle buckets for non-monotonic length sampling.
            (Default: True)
        drop_last (bool, optional): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
            (Default: False)
    Note:
        ``max_token_count`` and ``batch_size`` are mutually exclusive. Only one argument of the two
        should have value.
    Note:
        ``drop_last`` is only valid when ``batch_size`` argument is given.
    Note:
        if ``shuffle`` is True, it will only shuffle the data once. Please set ``reload_dataloaders_every_n_epochs=1``
        in pytorch_lightning Trainer to enable shuffling every epoch.
    """

    def __init__(
        self,
        lengths: List[int],
        num_buckets: int,
        frame_rate: int = 25,
        min_len: int = 0,
        max_len: Optional[int] = None,
        max_token_count: Optional[float] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if max_len is None:
            max_len = max(lengths)

        if not (0 <= min_len <= max_len):
            raise AssertionError("``min_len`` should be non-negative and smaller than ``max_len``")
        if max_token_count is not None and batch_size is not None:
            raise AssertionError("The ``max_token_count`` and ``batch_size`` can't be both set.")
        if max_token_count is None and batch_size is None:
            raise AssertionError("One of ``max_token_count`` or ``batch_size`` must be set.")

        # Convert max_token_count (seconds) to max_frames
        self.frame_rate = frame_rate
        if max_token_count is not None:
            max_token_count = int(max_token_count * frame_rate)
            assert (
                max_len <= max_token_count
            ), "The  ``max_token_count`` must be greater than or equal to the maximum value of ``lengths``."

        # Filter out samples which are outside the bounds of [min_len, max_len]
        filtered_length_idx = [(length, i) for i, length in enumerate(lengths) if min_len <= length <= max_len]
        if len(filtered_length_idx) == 0:
            raise AssertionError("``lengths`` cannot be empty after filtering.")
        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buckets = self._get_buckets(self.lengths, num_buckets, min_len, max_len)
        self._update_iter_list()

    def _get_buckets(self, lengths: List[int], num_buckets: int, min_len: int, max_len: int) -> Dict[int, Tensor]:
        """Generate buckets based on the dataset.
        Args:
            lengths (List[int]): The lengths of the samples in the dataset.
            num_buckets (int): The number of buckets.
            min_len (int): The lower bound of the evenly spaced length intervals to determine bucket width.
            max_len (int): The upper bound of the evenly spaced length intervals to determine bucket width.
        Returns:
            (dict[int, Tensor]): A dictionary in which the key is the bucket index, the value is
                the Tensor of corresponding sample indices.
        """
        buckets = {}
        boundaries = torch.linspace(min_len - 1, max_len + 1, num_buckets + 1)
        bucket_ids = torch.bucketize(torch.tensor(lengths), boundaries)
        for i in range(bucket_ids.size(0)):
            bucket_id = int(bucket_ids[i])
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        for k in buckets:
            buckets[k] = torch.as_tensor(buckets[k], dtype=torch.int)
        buckets = {k: v for k, v in sorted(buckets.items())}
        return buckets

    def _update_iter_list(self) -> None:
        if self.shuffle:
            for k in self.buckets:
                self.buckets[k] = self.buckets[k][torch.randperm(self.buckets[k].size(0))]
        self.iter_list = []
        total_len = 0
        batch = []
        max_batch_size = self.max_token_count if self.max_token_count else self.batch_size
        for k in self.buckets:
            for i in range(self.buckets[k].size(0)):
                index = int(self.buckets[k][i])
                sample_length = self.lengths[index] if self.max_token_count else 1
                if total_len + sample_length <= max_batch_size:
                    batch.append(self.indices[index])
                    total_len += sample_length
                else:
                    self.iter_list.append(batch)
                    batch = [self.indices[index]]
                    total_len = sample_length
        if len(batch) > 0 and (self.max_token_count or not self.drop_last):
            self.iter_list.append(batch)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.iter_list)

    def __len__(self):
        if self.batch_size or (self.max_token_count and not self.shuffle):
            return len(self.iter_list)


class DistributedBatchSampler(DistributedSampler):
    """`BucketizeBatchSampler` wrapper that distributes across each processor.
    Args:
        batch_sampler (BucketizeBatchSampler): the initialized bucketize batch sampler.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): if ``True``, the list of batch indices will be shuffled.
            (Default: ``True``)
        seed (int, optional): random seed used to shuffle the batch_sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. (Default: ``0``)
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. (Default: ``False``)
    Note:
        if ``shuffle`` is True, it will only shuffle the data once. Please set ``reload_dataloaders_every_n_epochs=1``
        in pytorch_lightning Trainer, and set `sampler.set_epoch(self.current_epoch)` before DataLoader initialization
        in `train_dataloader` method to enable shuffling every epoch.
    """

    def __init__(
        self,
        batch_sampler: BucketizeBatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_sampler = batch_sampler
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        if shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.batch_sampler.iter_list), generator=g).tolist()
            indices = [self.batch_sampler.iter_list[i] for i in perm]
        else:
            indices = self.batch_sampler.iter_list
        if self.drop_last:
            self.total_size = len(indices) - len(indices) % self.num_replicas
        else:
            padding_size = self.num_replicas - len(indices) % self.num_replicas
            indices += indices[:padding_size]
            self.total_size = len(indices)
        self.num_samples = self.total_size // self.num_replicas
        self.subset = indices[self.rank : self.total_size : self.num_replicas]
        assert len(self.subset) == self.num_samples

    def __iter__(self):
        subset = iter(self.subset)
        return subset

    def __len__(self):
        return self.num_samples

class VideoDataset(Dataset):
    """Create a Dataset for video data for training and fine-tuning.
    Args:
        tsv_dir (str or Path): The root directory of the ``.tsv`` file list.
        subset (str): The subset of the dataset. Options: [``train``, ``valid``].
    """

    def __init__(
        self,
        tsv_dir: Union[str, Path],
        subset: str,
    ) -> None:
        self.f_list, self.ind_list, self.len_list = self._get_lists(Path(tsv_dir), subset)

    def __len__(self):
        return len(self.f_list)

    def _get_lists(
        self,
        tsv_dir: Path,
        subset: str,
    ) -> Tuple[List[Path], List[int], List[int]]:
        """Get the list of paths for iteration.
        Args:
            tsv_dir (Path): The root directory of the ``.tsv`` file list.
            subset (str): The subset of the dataset. Options: [``train``, ``valid``].
        Returns:
            (numpy.array) List of file paths.
            (numpy.array) List of indices.
            (numpy.array) List of video frame lengths.
        """
        f_ind_len_list = []
        with open(tsv_dir / f"{subset}.tsv") as f:
            root = f.readline().rstrip()
            for index, line in enumerate(f):
                path, nframes = line.split("\t")
                path = f"{root}/{path}"
                nframes = int(nframes)
                f_ind_len_list.append((path, index, nframes))
        f_list, ind_list, len_list = zip(*f_ind_len_list)
        return np.asarray(f_list), np.asarray(ind_list), np.asarray(len_list)

    def _load_video(self, index: int) -> Tuple[Tensor, int]:
        """
        Load video data from .h5 file.
        rtype: torch.Tensor, T x C x H x W
        """
        path = self.f_list[index]  # .h5 파일 경로
        if path.endswith(".h5"):    
            with h5py.File(path, "r") as h5_file:
                vid_data = h5_file["data"][:]  # numpy 배열로 반환됩니다.
            vid_data = np.expand_dims(vid_data, axis=1)  # (T, H, W, C) -> (T, 1, H, W)
            return torch.tensor(vid_data), vid_data.shape[0]
        elif path.endswith(".mp4"):
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            vid_tensor = torch.tensor(np.array(frames)) # B x H x W x C , [75, 96, 96, 3]
            vid_tensor = vid_tensor.permute(0, 3, 1, 2)
            return vid_tensor, vid_tensor.size(0)

    def __getitem__(self, index):
        frames = self._load_video(index)  # (num_frames, height, width, channels)
        if type(frames) == torch.Tensor:
            length = frames.size(0)
        else: # tuple
            frames, length = frames
        return frames, length

def _crop_video(
    frames: Tensor,
    length: int,
    num_frames: int,
    rand_crop: bool,
) -> Tuple[Tensor, int]:
    """Crop the video frames.
    Args:
        frames (Tensor): The video frames Tensor with dimensions `(num_frames, height, width, channels)`.
        length (int): The total number of frames.
        num_frames (int): The desired number of frames after cropping.
        rand_crop (bool): if ``rand_crop`` is True, the starting index of the
            frames is random if the length is longer than the desired number of frames.
    Returns:
        (Tuple[Tensor, int]):
            - The cropped Tensor of frames.
            - The number of frames after cropping.
    """
    frame_offset = 0
    if frames.size(0) > num_frames and rand_crop:
        diff = frames.size(0) - num_frames
        frame_offset = torch.randint(diff, size=(1,))
    elif frames.size(0) < num_frames:
        num_frames = frames.size(0)
    frames = frames[frame_offset : frame_offset + num_frames]
    length = num_frames

    return frames, length

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)

class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                # torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        return self.video_pipeline(sample)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned

class CollateFnVideo:
    """The collate class for video data pre-training and fine-tuning.
    Args:
        pad (bool): If ``True``, the frames will be padded to the max length in the mini-batch.
            If ``pad`` is False, the frames will be cropped to the minimum length in the mini-batch.
            (Default: False)
        rand_crop (bool): if ``True``, the starting index of the frames is random
            if the length is longer than the minimum length in the mini-batch.
    """

    def __init__(
        self,
        pad: bool = False,
        rand_crop: bool = True,
        subset: str = "train",
    ) -> None:
        self.pad = pad
        self.rand_crop = rand_crop
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )


    def __call__(self, batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (List[Tuple(Tensor, int)]):
                The list of tuples that contains the video frames and frame lengths.
        Returns:
            (Tuple(Tensor, Tensor)):
                The Tensor of video frames with dimensions `(batch, frames, height, width, channels)`.
                The Tensor of frame lengths with dimension `(batch,)`.
        """
        if self.pad:
            num_frames = max([sample[0].shape[0] for sample in batch])
        else:
            num_frames = min([sample[0].shape[0] for sample in batch])

        frames_list, lengths = [], []
        for sample in batch:
            frames, length = sample  # frames: (frames, height, width, channels)
            # frames, length = _crop_video(frames, length, num_frames, self.rand_crop)  # frames: (num_frames, height, width, channels)
            frames_list.append(frames)
            lengths.append(length)

        # Ensure all samples have the same frame shape, apply padding if needed
        if self.pad:
            max_height = max([frames.shape[1] for frames in frames_list])
            max_width = max([frames.shape[2] for frames in frames_list])
            max_channels = frames_list[0].shape[3]  # Assume all videos have the same channel count

            padded_frames_list = []
            for frames in frames_list:
                pad_height = max_height - frames.shape[1]
                pad_width = max_width - frames.shape[2]
                padded_frames = torch.nn.functional.pad(
                    frames,
                    (0, 0, 0, pad_width, 0, pad_height),  # Pad height, width
                    value=0,  # Assume black padding
                )
                padded_frames_list.append(padded_frames)
            frames_list = padded_frames_list
        frames = torch.nn.utils.rnn.pad_sequence(
            [frames.clone().detach() for frames in frames_list], batch_first=True
        )
        frames = self.video_pipeline(frames) 
        lengths = torch.tensor(lengths)
        return frames, lengths
