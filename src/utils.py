import librosa
import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.ops.boxes import box_area
import os
from typing import Optional, List
from torch import Tensor
import random
import shutil

def _stft(audio, stft_frame=1022, stft_hop=256):
    spec = librosa.stft(audio, n_fft=stft_frame, hop_length=stft_hop)
    amp = np.abs(spec)
    phase = np.angle(spec)
    return amp, phase

def _load_audio_file(path):
    if path.endswith('.mp3'):
        audio_raw, rate = torchaudio.load(path)
        audio_raw = audio_raw.numpy().astype(np.float32)
        audio_raw *= (2.0**-31)
        if audio_raw.shape[1] == 2:
            audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
        else:
            audio_raw = audio_raw[:, 0]
    else:
        audio_raw, rate = librosa.load(path, sr=None, mono=True)
    return audio_raw, rate

def _load_audio(path, center_timestamp, audLen=65535, audSec=6, audRate=11025, mode='train', nearest_resample=False):
    audio = np.zeros(audLen, dtype=np.float32)

    if path.endswith('silent'):
        return audio

    audio_raw, rate = _load_audio_file(path)

    if audio_raw.shape[0] < rate * audSec:
        audio_raw = np.tile(audio_raw, int(rate * audSec / audio_raw.shape[0]) + 1)

    if rate > audRate:
        if nearest_resample:
            audio_raw = audio_raw[::rate//audRate]
        else:
            audio_raw = librosa.resample(audio_raw, rate, audRate)

    len_raw = audio_raw.shape[0]
    center = int(center_timestamp * audRate)
    start = max(0, center - audLen // 2)
    end = min(len_raw, center + audLen // 2)
    audio[audLen//2-(center-start): audLen//2+(end-center)] = audio_raw[start:end]

    if mode == 'train':
        scale = random.random() + 0.5     # 0.5-1.5
        audio *= scale
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def _mix_n_and_stft(audios, stft_frame, stft_hop):
    N = len(audios)
    audios = [audio / N for audio in audios]
    audio_mix = np.sum(audios, axis=0)

    amp_mix, phase_mix = _stft(audio_mix, stft_frame, stft_hop)
    mags = [np.expand_dims(_stft(audio, stft_frame, stft_hop)[0], axis=0) for audio in audios]
    
    return np.expand_dims(amp_mix, axis=0), mags, np.expand_dims(phase_mix, axis=0)

def generate_spectrogram_magphase(audio, stft_frame=1022, stft_hop=256, with_phase=True):
    spectro = librosa.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    return spectro_mag

def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['loss'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['loss'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['coseparation_loss'],
             color='g', label='coseparation')
    plt.plot(history['val']['epoch'], history['val']['bbox_loss'],
             color='b', label='bbox')
    plt.legend()
    fig.savefig(os.path.join(path, 'val_loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['coseparation_loss'],
             color='g', label='coseparation')
    plt.plot(history['train']['epoch'], history['train']['bbox_loss'],
             color='b', label='bbox')
    plt.legend()
    fig.savefig(os.path.join(path, 'train_loss.png'), dpi=200)
    plt.close('all')


def plot_pretrain_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['loss'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['loss'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['accuracy'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['accuracy'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'accuracy.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['recall'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['recall'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'recall.png'), dpi=200)
    plt.close('all')

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def warpgrid(bs, HO, WO, warp=True):
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    return grid.astype(np.float32)

def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)

def visualizeSpectrogram(spectrogram, save_path):
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    plt.pcolormesh(librosa.amplitude_to_db(spectrogram.squeeze()))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualizeBbox(frame_path, det_boxes, save_path, classes, rescale=False):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    colors = COLORS * 100
    pil_img = Image.open(frame_path)
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    ax.axis('off')
    clss, probs, boxes = det_boxes[:, 1], det_boxes[:, 2], torch.tensor(det_boxes[:, -4:])
    if rescale:
        boxes = rescale_bboxes(boxes, pil_img.size)
    for cl, p, (xmin, ymin, xmax, ymax), c in zip(clss, probs, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        ax.text(xmin, ymin, f'{classes[int(cl)]}: {p:.2f}', fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.savefig(save_path)
    plt.close()

def reduce_by_overlap(detections, overlap_thresh):
    if len(detections) <= 1:
        return detections
    x1min, y1min, x1max, y1max = detections[:, 3], detections[:, 4], detections[:, 5], detections[:, 6]
    areas = (x1max - x1min) * (y1max - y1min)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1min[i], x1min[order[1:]])
        yy1 = np.maximum(y1min[i], y1min[order[1:]])
        xx2 = np.minimum(x1max[i], x1max[order[1:]])
        yy2 = np.minimum(y1max[i], y1max[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
    return detections[keep]

def filter_detections(detection_npy, total_frames):
    total_cls = 15
    confidence_thresh = 0.7
    overlap_thresh = 0.7
    max_number_of_cls = 2
    single_detection_each_class = True
    
    # Filter by confidence and organize by frame
    confidence_mask = detection_npy[:, 2] > confidence_thresh
    filtered_detections = detection_npy[confidence_mask]

    frame_detections = [[] for _ in range(total_frames)]
    class_detections = [[] for _ in range(total_cls)]

    for det in filtered_detections:
        frame = int(det[0])
        cls = int(det[1])
        frame_detections[frame].append(det)
        class_detections[cls].append(det)

    # Reduce by overlap for each frame
    for frame in range(total_frames):
        if frame_detections[frame]:
            frame_detections[frame] = reduce_by_overlap(np.array(frame_detections[frame]), overlap_thresh)
    
    # Organize detections by class after overlap reduction
    class_detections = [[] for _ in range(total_cls)]
    for frame_det in frame_detections:
        for det in frame_det:
            cls = int(det[1])
            class_detections[cls].append(det)

    # Single detection for each class
    if single_detection_each_class:
        for cls in range(total_cls):
            if class_detections[cls]:
                best_det = max(class_detections[cls], key=lambda x: x[2])
                class_detections[cls] = [best_det]

    # Collect all detections and keep top N classes with highest confidence
    all_detections = [(cls, det) for cls, dets in enumerate(class_detections) for det in dets]
    if len(all_detections) > max_number_of_cls:
        all_detections = sorted(all_detections, key=lambda x: x[1][2], reverse=True)[:max_number_of_cls]

    # Combine and return indices of top detections
    top_indices = [np.where((detection_npy == det).all(axis=1))[0][0] for _, det in all_detections]
    return top_indices


def _max_by_axis(the_list):
    return [max(items) for items in zip(*the_list)]

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device, non_blocking=True)
        cast_mask = self.mask.to(device, non_blocking=True) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_shape[0], batch_shape[2], batch_shape[3]), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)

class AverageMeter:
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return 0. if self.val is None else self.val.tolist()

    def average(self):
        return 0. if self.avg is None else self.avg.tolist()

