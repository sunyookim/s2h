#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from src.options import Options
from src.dataset import CreateDatasetDataloader
from src.models import CreateModel
from src.utils import visualizeSpectrogram, visualizeBbox, plot_pretrain_metrics, plot_loss_metrics, generate_spectrogram_magphase, AverageMeter, istft_reconstruction, warpgrid, makedirs, NestedTensor
from src import criterion
from src.criterion import HungarianMatcher, SetCriterion
from mir_eval.separation import bss_eval_sources
import soundfile as sf
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Calculate metrics
def calc_metrics(batch_data, outputs, opt):
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    batch_size = opt.batch_size
    num_mix = opt.num_mix
    mag_mix = batch_data['mag_mix'].cpu().numpy()  # (8, 1, 512, 256)
    phase_mix = batch_data['phase_mix'].cpu().numpy()  # (8, 1, 512, 256)
    audios = batch_data['audios'].cpu().numpy()  # (8, 2, 65535)
    pred_masks = outputs['pred_mask']  # [b][n]->[1, 256, 256]

    pred_masks_linear = []
    for b in range(batch_size):
        batch_masks = []
        for n in range(num_mix):
            mask = pred_masks[b][n].unsqueeze(0)
            warp_grid = torch.from_numpy(warpgrid(1, opt.stft_frame // 2 + 1, 256, False)).to(opt.device)
            sampled_mask = F.grid_sample(mask, warp_grid, align_corners=True).cpu().numpy()
            batch_masks.append(sampled_mask.squeeze(0))
        pred_masks_linear.append(batch_masks)

    for b in range(batch_size):
        gts_wav, preds_wav = [], []
        mix_wav = istft_reconstruction(mag_mix[b].squeeze(0), phase_mix[b].squeeze(0), hop_length=opt.stft_hop)
        for n in range(num_mix):
            pred_mag = mag_mix[b] * pred_masks_linear[b][n]
            pred_wav = istft_reconstruction(pred_mag.squeeze(0), phase_mix[b].squeeze(0), hop_length=opt.stft_hop)
            preds_wav.append(pred_wav)

        valid = True
        for n in range(num_mix):
            gt_wav = audios[b][n][:len(preds_wav[0])]
            gts_wav.append(gt_wav)
            valid &= np.sum(np.abs(gts_wav[n])) > 1e-5 and np.sum(np.abs(preds_wav[n])) > 1e-5

        if valid:
            sdr, sir, sar, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray(preds_wav), False)
            sdr_mix, _, _, _ = bss_eval_sources(np.asarray(gts_wav), np.asarray([mix_wav[:len(preds_wav[0])] for _ in range(num_mix)]), False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(), sdr_meter.average(), sir_meter.average(), sar_meter.average()]

# Visualize predictions
def output_visuals(batch_data, outputs, opt):
    batch_size = opt.batch_size
    num_mix = opt.num_mix

    mag_mix = batch_data['mag_mix'].cpu().numpy()
    phase_mix = batch_data['phase_mix'].cpu().numpy()
    audios = batch_data['audios'].cpu().numpy()
    path_frames = batch_data['path_frames']
    path_audios = batch_data['path_audios']

    pred_masks = outputs['pred_mask']
    detections = outputs['detections']

    classes = ["Banjo", "Cello", "Drum", "Guitar", "Harp", "Harmonica", "Oboe", "Piano", "Saxophone", "Trombone", "Trumpet", "Violin", "Flute", "Accordion", "French horn"] \
        if opt.detr_dataset == 'openimages_instruments' else \
        ["null", "accordion", "acoustic_guitar", "cello", "clarinet", "erhu", "flute", "saxophone", "trumpet", "tuba", "violin", "xylophone"]

    pred_masks_linear = []
    for b in range(batch_size):
        batch_masks = []
        for n in range(num_mix):
            mask = pred_masks[b][n].unsqueeze(0)
            warp_grid = torch.from_numpy(warpgrid(1, opt.stft_frame // 2 + 1, 256, False)).to(opt.device)
            sampled_mask = F.grid_sample(mask, warp_grid, align_corners=True).cpu().numpy()
            batch_masks.append(sampled_mask.squeeze(0))
        pred_masks_linear.append(batch_masks)

        prefix = '+'.join(['-'.join(path_audios[b][n].split('/')[-2:]).split('.')[0] for n in range(num_mix)])
        output_dir = os.path.join(opt.vis, prefix)
        makedirs(output_dir, remove=True)

        gt_audios = [audios[b][n] for n in range(num_mix)]
        mixed_audios = np.sum(gt_audios, 0)
        sf.write(os.path.join(output_dir, 'audio_mixed.wav'), mixed_audios, opt.audRate)
        audio_mix_mag = generate_spectrogram_magphase(mixed_audios, opt.stft_frame, opt.stft_hop, with_phase=False)
        visualizeSpectrogram(audio_mix_mag[0, :, :], os.path.join(output_dir, 'audio_mixed_spec.png'))

        for n in range(num_mix):
            sf.write(os.path.join(output_dir, f'audio{n+1}_gt.wav'), gt_audios[n], opt.audRate)
            gt_mag = generate_spectrogram_magphase(gt_audios[n], opt.stft_frame, opt.stft_hop, with_phase=False)
            visualizeSpectrogram(gt_mag[0, :, :], os.path.join(output_dir, f'audio{n+1}_spec.png'))

            pred_mag = mag_mix[b] * pred_masks_linear[b][n]
            pred_wav = istft_reconstruction(pred_mag.squeeze(0), phase_mix[b].squeeze(0), hop_length=opt.stft_hop)
            sf.write(os.path.join(output_dir, f'separation{n+1}.wav'), pred_wav, opt.audRate)
            pred_mag = generate_spectrogram_magphase(pred_wav, opt.stft_frame, opt.stft_hop, with_phase=False)
            visualizeSpectrogram(pred_mag[0, :, :], os.path.join(output_dir, f'separation{n+1}_spec.png'))

            frame_path = path_frames[b][n][int(detections[b][n][0][0])]
            det_boxes = detections[b][n]
            visualizeBbox(frame_path, det_boxes, os.path.join(output_dir, f'audio{n+1}_det.png'), classes=classes, rescale=True)

def get_coseparation_loss(output, loss_coseparation):
    predicted_masks = output['pred_mask']  # list of lists of tensors
    gt_masks = output['gt_mask']  # list of lists of tensors
    weights = output['weight']  # tensor
    predicted_masks_sum = torch.sum(predicted_masks, dim=2, keepdim=True)
    coseparation_loss = loss_coseparation(predicted_masks_sum, gt_masks, weights)
    return coseparation_loss

def get_bbox_loss(batch_data, output, opt):
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    setcriterion = SetCriterion(opt.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'boxes']).to(opt.device)
    targets = (batch_data['labels'], batch_data['boxes'], batch_data['sizes'])
    loss_dict = setcriterion(output, targets)
    bbox_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / sum(weight_dict.values())
    return bbox_loss


# Training function
def train(model, loader, optimizer, history, epoch, crit, opt):
    opt.mode = 'train'
    torch.set_grad_enabled(True)
    model.train()

    with tqdm(total=len(loader), desc=f'Epoch {epoch} [Train]') as pbar:
        for i, batch_data in enumerate(loader):
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(opt.device, non_blocking=True)
                elif isinstance(value, NestedTensor):
                    batch_data[key] = value.to(opt.device)

            optimizer.zero_grad()

            output = model(batch_data, mode=opt.mode)
            coseparation_loss = get_coseparation_loss(output, opt, crit['loss_coseparation']) * opt.coseparation_loss_weight
            bbox_loss = get_bbox_loss(batch_data, output, opt) * opt.bbox_loss_weight
            loss = coseparation_loss + bbox_loss

            loss.backward()  
            optimizer.step()  # Perform optimizer step

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            if i % opt.disp_iter == 0:
                fractional_epoch = epoch - 1 + 1. * i / opt.epoch_iters
                history['train']['epoch'].append(fractional_epoch)
                history['train']['loss'].append(loss.item())
                history['train']['coseparation_loss'].append(coseparation_loss.item())
                history['train']['bbox_loss'].append(bbox_loss.item())

    # Ensure model is set to training mode
    model.train()

def evaluate(model, loader, history, epoch, crit, opt):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    makedirs(opt.vis, remove=True)
    model.eval()

    loss_meter = AverageMeter()
    coseparation_loss_meter = AverageMeter()
    bbox_loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    with tqdm(total=len(loader), desc=f'Epoch {epoch} [Eval]') as pbar:
        for i, batch_data in enumerate(loader):
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(opt.device, non_blocking=True)
                elif isinstance(value, NestedTensor):
                    batch_data[key] = value.to(opt.device)

            output = model.forward(batch_data, mode=opt.mode)
            coseparation_loss = get_coseparation_loss(output, crit['loss_coseparation']) * opt.coseparation_loss_weight
            bbox_loss = get_bbox_loss(batch_data, output, opt) * opt.bbox_loss_weight
            loss = coseparation_loss + bbox_loss

            loss_meter.update(loss.item())
            coseparation_loss_meter.update(coseparation_loss.item())
            bbox_loss_meter.update(bbox_loss.item())
            print('[Eval] iter {}, loss: {:.4f}'.format(i, loss.item()))

            sdr_mix, sdr, sir, sar = calc_metrics(batch_data, output, opt)

            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)

            if opt.output_visuals:
                output_visuals(batch_data, output, opt)
                pbar.update(1)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    
    if opt.mode != 'test':
        file_name = os.path.join(opt.checkpoints_dir, f'results.txt')
    else:
        file_name = os.path.join(opt.checkpoints_dir, f'test_results.txt')
    with open(file_name, 'wt') as file:
        file.write('SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
                   .format(sdr_meter.average(),
                           sir_meter.average(),
                           sar_meter.average()))
    
    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['coseparation_loss'].append(coseparation_loss_meter.average())
    history['val']['bbox_loss'].append(bbox_loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(opt.checkpoints_dir, history)

def checkpoint(model, history, epoch, opt):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history, '{}/history_{}'.format(opt.checkpoints_dir, suffix_latest))
    torch.save(model.module.state_dict(), '{}/model_{}'.format(opt.checkpoints_dir, suffix_latest))

    cur_loss = history['val']['loss'][-1]
    cur_sdr = history['val']['sdr'][-1]
    if cur_sdr > opt.best_sdr:
        opt.best_sdr = cur_sdr
        torch.save(model.module.state_dict(), '{}/model_{}'.format(opt.checkpoints_dir, suffix_best))

def create_optimizer(model, opt):
    param_groups = [
        {'params': [p for n, p in model.net_visual.named_parameters() if "backbone" in n and p.requires_grad], 'lr': opt.lr_backbone},
        {'params': [p for n, p in model.net_visual.named_parameters() if "backbone" not in n and p.requires_grad], 'lr': opt.lr_visual},
        {'params': model.net_audiovisual.encoder.parameters(), 'lr': opt.lr_audio},
        {'params': model.net_audiovisual.decoder.parameters(), 'lr': opt.lr_audiovisual},
        {'params': model.net_audiovisual.unet_decoder.parameters(), 'lr': opt.lr_unet}
    ]
    return torch.optim.Adam(param_groups)

def adjust_learning_rate(optimizer, opt):
    opt.lr_backbone *= 0.1
    opt.lr_visual *= 0.1
    opt.lr_audio *= 0.1
    opt.lr_audiovisual *= 0.1
    opt.lr_unet *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def main(opt):
    model = CreateModel(opt)
    optimizer = create_optimizer(model, opt)

    model = torch.nn.DataParallel(model)
    model.to(opt.device)    

    if opt.mode == 'pretrain':
        dataset_train, loader_train = CreateDatasetDataloader(opt)
        opt.mode = 'pretrain_val'
        dataset_val, loader_val = CreateDatasetDataloader(opt)
        opt.mode = 'pretrain'
        opt.epoch_iters = len(dataset_train) // opt.batch_size
        print('1 Epoch = {} iters'.format(opt.epoch_iters))
        
        history = {
            'train': {'epoch': [], 'loss': [], 'accuracy': [], 'recall': []},
            'val': {'epoch': [], 'loss': [], 'accuracy': [], 'recall': []}}

        for epoch in range(1, opt.epochs + 1):
            torch.set_grad_enabled(True)
            model.train()
            opt.mode = 'pretrain'
            with tqdm(total=len(loader_train), desc=f'Epoch {epoch} [Train]') as pbar:
                for i, batch_data in enumerate(loader_train):
                    for key, value in batch_data.items():
                        if isinstance(value, torch.Tensor):
                            batch_data[key] = value.to(opt.device, non_blocking=True)
                        elif isinstance(value, NestedTensor):
                            batch_data[key] = value.to(opt.device)

                    optimizer.zero_grad()

                    loss, accuracy, recall = model.module.pretrain(batch_data, mode=opt.mode)

                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), recall=recall.item())
                    pbar.update(1)

                    fractional_epoch = epoch - 1 + 1. * i / opt.epoch_iters
                    history['train']['epoch'].append(fractional_epoch)
                    history['train']['loss'].append(loss.item())
                    history['train']['accuracy'].append(accuracy.item())
                    history['train']['recall'].append(recall.item())

            print('Evaluating at {} epochs...'.format(epoch))
            torch.set_grad_enabled(False)
            model.eval()
            opt.mode = 'pretrain_val'

            loss_meter = AverageMeter()
            accuracy_meter = AverageMeter()
            recall_meter = AverageMeter()

            with tqdm(total=len(loader_val), desc=f'Epoch {epoch} [Eval]') as pbar:
                for i, batch_data in enumerate(loader_val):
                    for key, value in batch_data.items():
                        if isinstance(value, torch.Tensor):
                            batch_data[key] = value.to(opt.device, non_blocking=True)
                        elif isinstance(value, NestedTensor):
                            batch_data[key] = value.to(opt.device)

                    loss, accuracy, recall = model.module.pretrain(batch_data, mode=opt.mode)

                    loss_meter.update(loss.item())
                    accuracy_meter.update(accuracy.item())
                    recall_meter.update(recall.item())
                    pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), recall=recall.item())
                    pbar.update(1)

            print('[Eval Summary] Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}'
                .format(epoch, loss_meter.average(), accuracy_meter.average(), recall_meter.average()))

            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(loss_meter.average())
            history['val']['accuracy'].append(accuracy_meter.average())
            history['val']['recall'].append(recall_meter.average())

            if epoch > 0:
                print('Plotting figures...')
                plot_pretrain_metrics(opt.checkpoints_dir, history)

            print('Saving checkpoints at {} epochs.'.format(epoch))
            suffix_latest = 'latest.pth'
            suffix_best = 'best.pth'

            torch.save(history, '{}/history_{}'.format(opt.checkpoints_dir, suffix_latest))
            torch.save(model.module.state_dict(), '{}/model_{}'.format(opt.checkpoints_dir, suffix_latest))

            cur_loss = history['val']['loss'][-1]

            if cur_loss > opt.best_loss:
                opt.best_loss = cur_loss
                torch.save(model.module.state_dict(), '{}/model_{}'.format(opt.checkpoints_dir, suffix_best))

            if epoch in opt.lr_steps:
                adjust_learning_rate(optimizer, opt)

    else:
        loss_coseparation = criterion.L1Loss().to(opt.device)
        loss_bbox = (criterion.L1Loss().to(opt.device), criterion.CELoss().to(opt.device))
        crit = {'loss_coseparation': loss_coseparation, 'loss_bbox': loss_bbox}

        if opt.mode == 'train':
            dataset_train, loader_train = CreateDatasetDataloader(opt)
            opt.mode = 'val'
            dataset_val, loader_val = CreateDatasetDataloader(opt)
            opt.mode = 'train'
            opt.epoch_iters = len(dataset_train) // opt.batch_size
            print('1 Epoch = {} iters'.format(opt.epoch_iters))
            
        elif opt.mode == 'test':
            dataset_test, loader_test = CreateDatasetDataloader(opt)

        history = {
            'train': {'epoch': [], 'loss': [], 'coseparation_loss': [], 'bbox_loss': []},
            'val': {'epoch': [], 'loss': [], 'coseparation_loss': [], 'bbox_loss': [], 'sdr': [], 'sir': [], 'sar': []}}

        if opt.mode == 'test':
            evaluate(model, loader_test, history, 0, crit, opt)
            print('Evaluation Done!')
            return

        for epoch in range(1, opt.epochs + 1):
            train(model, loader_train, optimizer, history, epoch, crit, opt)
            opt.mode = 'val'
            evaluate(model, loader_val, history, epoch, crit, opt)
            checkpoint(model, history, epoch, opt)

            if epoch in opt.lr_steps:
                adjust_learning_rate(optimizer, opt)

        print('Training Done!')

if __name__ == '__main__':
    opt = Options().parse()
    opt.device = torch.device("cuda")
    
    opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.id)

    if opt.mode == 'pretrain':
        makedirs(opt.checkpoints_dir, remove=True)
        opt.lr_audiovisual = 0
        opt.lr_unet = 0
    elif opt.mode == 'train':
        makedirs(opt.checkpoints_dir, remove=True)
        opt.vis = os.path.join(opt.checkpoints_dir, 'visualization/')
    else:
        opt.weights = os.path.join(opt.checkpoints_dir, 'model_best.pth')
        opt.vis = os.path.join(opt.checkpoints_dir, 'test_visualization/')
    
    if opt.detr_dataset == 'openimages_instruments':
        opt.num_queries = 16
        opt.num_classes = 15
    elif opt.detr_dataset == 'roboflow_instruments':
        opt.num_queries = 12
        opt.num_classes = 12
    
    if opt.pretrain_encoders and opt.mode == 'train':
        opt.weights = os.path.join('pretrained_models/ast_detr', opt.detr_dataset, 'checkpoint_ept.pth')
    
    file_name = os.path.join(opt.checkpoints_dir, f'opt_{opt.mode}.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(vars(opt).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    opt.best_loss = float("inf")
    opt.best_sdr = float("-inf")

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    main(opt)
