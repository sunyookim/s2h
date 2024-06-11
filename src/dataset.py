import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import csv
from torchvision import transforms
from src.utils import box_xyxy_to_cxcywh, _load_audio, _stft, _mix_n_and_stft, nested_tensor_from_tensor_list
from joblib import Parallel, delayed

class DETRFeatureDataset(Dataset):
    def __init__(self, opt):
        self.num_mix = opt.num_mix
        self.num_frames = opt.num_frames
        self.max_width = opt.max_width
        self.stride_frames = opt.stride_frames
        self.fps = opt.frameRate
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate

        # STFT params
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop

        self.mode = opt.mode
        self.seed = opt.seed

        self.data_dir = opt.data_dir
        self.dataset = opt.dataset
        self.detr_dataset = opt.detr_dataset
        random.seed(self.seed)

        self.vid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        list_sample_path = os.path.join(self.data_dir, self.dataset, f'{self.mode}.csv')
        
        self.list_sample = []
        with open(list_sample_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            self.list_sample = [row for row in reader if len(row) >= 2]

        if self.mode in ['pretrain', 'train']:
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if self.mode in ['val', 'test', 'pretrain_val']:
            self.pairs = [(i, j) for i in range(len(self.list_sample)) for j in range(i + 1, len(self.list_sample))]

    def __len__(self):
        if self.mode in ['val', 'test', 'pretrain_val']:
            return len(self.pairs)
        else:
            return len(self.list_sample)

    def get_classes_from_path(self, path):
        parts = path.split('/')
        if 'solo' in parts:
            return [parts[parts.index('solo') + 2]]
        elif 'duet' in parts:
            return parts[parts.index('duet') + 2].split('+')
        return []

    def __getitem__(self, index):
        if self.mode in ['pretrain', 'pretrain_val']:
            return self._load_pretrain_item(index)
        elif self.mode == 'train':
            return self._load_train_item(index)
        elif self.mode in ['val', 'test']:
            return self._load_val_test_item(index)

    def _load_pretrain_item(self, index):
        if self.mode == 'pretrain_val':
            index1, index2 = self.pairs[index]
            path_audio1, path_frame1, count_frames1 = self.list_sample[index1]
            path_audio2, path_frame2, count_frames2 = self.list_sample[index2]
        else:
            path_audio1, path_frame1, count_frames1 = self.list_sample[index]
            index2 = random.randint(0, len(self.list_sample) - 1)
            path_audio2, path_frame2, count_frames2 = self.list_sample[index2]

        # Load first audio and frame
        frame_idx1 = random.randint(1, int(count_frames1))
        frame_path1 = os.path.join(self.data_dir, self.dataset, path_frame1, '{:06d}.jpg'.format(frame_idx1))
        audio_path1 = os.path.join(self.data_dir, self.dataset, path_audio1)
        center_time1 = (frame_idx1 - 0.5) / self.fps

        try:
            audio1 = _load_audio(audio_path1, center_time1, self.audLen, self.audSec, self.audRate, self.mode)
        except Exception as e:
            print(f'Failed loading pretrain audio for {audio_path1}: {e}')
            audio1 = np.zeros(self.audLen, dtype=np.float32)
        try:
            img1 = Image.open(frame_path1).convert('RGB')
            img1 = self._resize_image(img1)
            frame1 = self.vid_transform(img1)
        except Exception as e:
            print(f'Failed to load frames for {frame_path1}: {e}')
            img1 = torch.zeros(3, 224, 224)
        instrument_categories1 = self.get_classes_from_path(path_audio1)  # Get instrument categories

        # Load second sample
        frame_idx2 = random.randint(1, int(count_frames2))
        frame_path2 = os.path.join(self.data_dir, self.dataset, path_frame2, '{:06d}.jpg'.format(frame_idx2))
        audio_path2 = os.path.join(self.data_dir, self.dataset, path_audio2)
        center_time2 = (frame_idx2 - 0.5) / self.fps

        try:
            audio2 = _load_audio(audio_path2, center_time2, self.audLen, self.audSec, self.audRate, self.mode)
        except Exception as e:
            print(f'Failed loading pretrain audio for {audio_path2}: {e}')
            audio2 = np.zeros(self.audLen, dtype=np.float32)
        instrument_categories2 = self.get_classes_from_path(path_audio2)  # Get instrument categories

        # Mix the audio
        mixed_audio = 0.5 * audio1 + 0.5 * audio2

        # Compute STFT of the mixed audio
        amp_mix, _ = _stft(mixed_audio, self.stft_frame, self.stft_hop)
        mag_mix = torch.from_numpy(amp_mix)

        # Combine instrument categories
        mixed_instrument_categories = list(set(instrument_categories1 + instrument_categories2))

        return {'frame': frame1, 'mag': mag_mix, 'instruments': mixed_instrument_categories}

    def _load_train_item(self, index):
        N = self.num_mix
        infos = [self.list_sample[index]]
        class_list = self.get_classes_from_path(infos[0][0])

        for _ in range(1, N):
            while True:
                sample = random.choice(self.list_sample)
                sample_classes = self.get_classes_from_path(sample[0])
                if not any(cls in class_list for cls in sample_classes):
                    class_list.extend(sample_classes)
                    infos.append(sample)
                    break

        return self._process_infos(infos)

    def _load_val_test_item(self, index):
        index1, index2 = self.pairs[index]
        infos = [self.list_sample[index1], self.list_sample[index2]]
        return self._process_infos(infos)

    def _resize_image(self, img):
        w, h = img.size
        if w > self.max_width:
            aspect_ratio = h / w
            new_w = self.max_width
            new_h = int(new_w * aspect_ratio)
            img = img.resize((new_w, new_h), Image.ANTIALIAS)
        return img

    def _process_infos(self, infos):
        N = len(infos)
        frames = [[] for _ in range(N)]
        audios = [None for _ in range(N)]
        targets = [[] for _ in range(N)]
        path_frames = [[] for _ in range(N)]
        path_audios = ['' for _ in range(N)]
        path_detections = ['' for _ in range(N)]
        center_frames = [0 for _ in range(N)]

        idx_margin = max(int(self.fps * 1), (self.num_frames // 2) * self.stride_frames)
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            center_frameN = random.randint(idx_margin + 1, int(count_framesN) - idx_margin) if self.mode == 'train' else int(count_framesN) // 2 + 1
            center_frames[n] = center_frameN

            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                frame_path = os.path.join(self.data_dir, self.dataset, path_frameN, '{:06d}.jpg'.format(center_frameN + idx_offset))
                path_frames[n].append(frame_path)

            path_audios[n] = os.path.join(self.data_dir, self.dataset, path_audioN)
            path_detections[n] = os.path.join(self.data_dir, self.dataset, path_frameN.replace('frames', f'detections/{self.detr_dataset}/detr_top_detections') + '.npy')

        results = Parallel(n_jobs=-1)(
            delayed(self._load_data)(n, path_detections, path_frames, path_audios, center_frames) for n in range(N)
        )
        
        for n, result in enumerate(results):
            frames[n], audios[n], targets[n] = result

        mag_mix, mags, phase_mix = _mix_n_and_stft(audios, self.stft_frame, self.stft_hop)
        audios = torch.stack([torch.from_numpy(audio) for audio in audios])
        mags = torch.stack([torch.from_numpy(mag) for mag in mags])
        mag_mix = torch.from_numpy(mag_mix)
        phase_mix = torch.from_numpy(phase_mix)

        flattened_frames = [frame for frame_set in frames for frame in frame_set]

        ret_dict = {'mag_mix': mag_mix, 'frames': flattened_frames, 'mags': mags, 'targets': targets}
        if self.mode in ['val', 'test']:
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['path_frames'] = path_frames
            ret_dict['path_audios'] = path_audios
        return ret_dict

    def _load_data(self, n, path_detections, path_frames, path_audios, center_frames):
        vid_frames = []
        vid_targets = []

        try:
            detections = np.load(path_detections[n])
            for i in range(self.num_frames):
                img = Image.open(path_frames[n][i]).convert('RGB')
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                frame_detections = detections[detections[:, 0] == int(center_frames[n] + idx_offset)]
                gt_boxes = torch.Tensor(frame_detections[:, 3:])
                boxes = box_xyxy_to_cxcywh(gt_boxes) / torch.tensor([img.width, img.height, img.width, img.height], dtype=torch.float32)
                gt_labels = torch.LongTensor(frame_detections[:, 1])
                vid_targets.append({'boxes': boxes, 'labels': gt_labels})
        except Exception as e:
            detections = np.empty((0, 6))  # Empty detections
            vid_targets.append({'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.long)})
            print(f'Failed to load detections for {path_detections[n]}: {e}')

        for i in range(self.num_frames):
            try:
                img = Image.open(path_frames[n][i]).convert('RGB')
                img = self._resize_image(img)
                vid_frames.append(self.vid_transform(img))
            except Exception as e:
                print(f'Failed to load frames for {path_frames[n][i]}: {e}')
                vid_frames.append(torch.zeros(3, 224, 224))

        try:
            center_timeN = (center_frames[n] - 0.5) / self.fps
            audio = _load_audio(path_audios[n], center_timeN, self.audLen, self.audSec, self.audRate, self.mode)
        except Exception as e:
            print(f'Failed to load audio for {path_audios[n]}: {e}')
            audio = np.zeros(self.audLen, dtype=np.float32)

        return vid_frames, audio, vid_targets

def collate_fn(batch):
    batch_dict = {}
    keys = batch[0].keys()
    for key in keys:
        if key == 'frames':
            all_frames = [frame for sample in batch for frame in sample['frames']]
            batch_dict[key] = nested_tensor_from_tensor_list(all_frames)
        elif key == 'targets':
            all_boxes = [target['boxes'] for sample in batch for target_list in sample['targets'] for target in target_list]
            all_labels = [target['labels'] for sample in batch for target_list in sample['targets'] for target in target_list]
            all_sizes = [len(target['boxes']) for sample in batch for target_list in sample['targets'] for target in target_list]
            batch_dict['boxes'] = torch.cat(all_boxes)
            batch_dict['labels'] = torch.cat(all_labels)
            batch_dict['sizes'] = all_sizes
        elif key in ['path_frames', 'path_audios', 'instruments']:
            batch_dict[key] = [d[key] for d in batch]
        elif key == 'frame':
            all_frames = [sample['frame'] for sample in batch]
            batch_dict[key] = nested_tensor_from_tensor_list(all_frames)
        else:
            batch_dict[key] = torch.stack([d[key] for d in batch])
    return batch_dict

def CreateDatasetDataloader(opt):
    dataset = DETRFeatureDataset(opt)
    shuffle = True if opt.mode in ['train', 'pretrain'] else False
    drop_last = True if opt.mode != 'test' else False
    batch_size=opt.batch_size if opt.mode in ['train', 'pretrain'] else 55
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    print(f"Created {opt.mode} dataset and dataloader")
    print(f'# {opt.mode} images = {len(dataset)}')
    return dataset, dataloader
