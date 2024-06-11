import os
import copy
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.utils import warpgrid, filter_detections, NestedTensor
from audiost.src.models import ASTModel 

class ASTDETRModel(torch.nn.Module):
    def __init__(self, opt, net_visual, net_audiovisual):
        super().__init__()
        self.opt = opt
        self.net_visual = net_visual
        self.net_audiovisual = net_audiovisual
        if opt.model_size == 'small':
            self.net_visual.transformer.encoder.layers = nn.Sequential(*list(self.net_visual.transformer.encoder.layers.children())[:3])
            self.net_visual.transformer.decoder.layers = nn.Sequential(*list(self.net_visual.transformer.decoder.layers.children())[:3])
            self.net_audiovisual.encoder.blocks = nn.Sequential(*list(self.net_audiovisual.encoder.blocks.children())[:3])
            self.net_audiovisual.decoder.layers = nn.Sequential(*list(self.net_audiovisual.decoder.layers.children())[:3])
        if opt.freeze_encoders:
            for p in self.net_visual.backbone.parameters():
                p.requires_grad_(False)
            for n, p in self.net_visual.transformer.encoder.named_parameters():
                if 'projection_layer' not in n:
                    p.requires_grad_(False)
            for p in self.net_audiovisual.encoder.parameters():
                p.requires_grad_(False)

    def pretrain(self, input, mode='pretrain'):
        hidden_dim = self.opt.hidden_dim
        batch_size = self.opt.batch_size if mode == 'pretrain' else 55 
        
        # Visual network forward pass
        frames = input['frame']
        features, pos = self.net_visual.backbone(frames)
        src, mask = features[-1].decompose()
        memory = self.net_visual.transformer(self.net_visual.input_proj(src), mask, self.net_visual.query_embed.weight, pos[-1])[1] # [batch_size, hidden_dim, height, width]
        video_features = torch.mean(memory.view(batch_size, hidden_dim, -1), dim=-1) # [batch_size, hidden_dim]
        
        # Audio network forward pass
        mags = input['mag'].unsqueeze(1) + 1e-10 # [batch_size, 1, 512, 256]
        grid_warp = torch.from_numpy(warpgrid(mags.size(0), 256, 256, warp=True)).to(self.opt.device)
        mags = F.grid_sample(mags, grid_warp)  # [B, 1, 256, 256]
        x = torch.log(mags) #[batch_size, 1, 512, 256]
        x = x.transpose(2, 3) 
        x = self.net_audiovisual.encoder.patch_embed(x) #[batch_size, 256, 768]
        x = x + self.net_audiovisual.encoder.pos_embed[:, 2:]
        x = self.net_audiovisual.encoder.pos_drop(x)
        for blk in self.net_audiovisual.encoder.blocks:
            x = blk(x)
        x = self.net_audiovisual.encoder.norm(x) #[batch_size, 258, 768]
        memory = self.net_audiovisual.encoder.projection_layer(x) #[batch_size, 256, 256]
        audio_features = torch.mean(memory, dim=1) #[batch_size, 256]

        # # Normalize the features
        # video_features = F.normalize(video_features, p=2, dim=1)
        # audio_features = F.normalize(audio_features, p=2, dim=1)

        # Calculating the contrastive loss
        logits = torch.mm(video_features, audio_features.T)  # [batch_size, batch_size]

        # Generate positive/negative pair labels
        instrument_categories = input['instruments']
        labels = torch.zeros(batch_size, batch_size).float().to(logits.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if set(instrument_categories[i]).intersection(set(instrument_categories[j])):
                    labels[i, j] = 1

        # Contrastive loss
        pos_weight = 2.0
        neg_weight = 1.0
        weights = torch.where(labels == 1, pos_weight, neg_weight)

        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)

        # Calculate accuracy
        with torch.no_grad():
            preds = torch.sigmoid(logits) >= 0.5
            correct = (preds == labels).float()
            accuracy = correct.sum() / correct.numel()

        # Calculate recall
        true_positives = ((preds == 1) & (labels == 1)).float().sum()
        false_negatives = ((preds == 0) & (labels == 1)).float().sum()
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add small value to prevent division by zero

        return loss, accuracy, recall

    def forward(self, input, mode='train'):
        batch_size = self.opt.batch_size if mode == 'train' else 55 
        num_mix = self.opt.num_mix
        num_frames = self.opt.num_frames
        num_queries = self.opt.num_queries
        num_classes = self.opt.num_classes
        
        visual_features = []
        detections = [[None for n in range(num_mix)] for b in range(batch_size)]
        output_pred_boxes = []
        output_pred_logits = []
        output_video_queries = []

        frames_tensor = input['frames'].tensors  # [B*N*F, 3, H, W]
        mask = input['frames'].mask  # [B*N*F, H, W]

        chunk_size = batch_size * num_mix * num_frames // 2
        total_frames = frames_tensor.shape[0]

        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = frames_tensor[start:end]  # [chunk_size, 3, H, W]
            chunk_mask = mask[start:end]  # [chunk_size, H, W]

            output = self.net_visual(NestedTensor(chunk, chunk_mask))

            output_pred_logits.append(output['pred_logits'])
            output_pred_boxes.append(output['pred_boxes'])
            output_video_queries.append(output['video_queries'])

        output_pred_logits = torch.cat(output_pred_logits) 
        pred_logits = output_pred_logits.reshape(batch_size, num_mix, num_frames * num_queries, num_classes+1).detach()
        
        output_pred_boxes = torch.cat(output_pred_boxes)
        pred_boxes = output_pred_boxes.reshape(batch_size, num_mix, num_frames * num_queries, 4).detach()
        
        output_video_queries = torch.cat(output_video_queries)
        video_queries = output_video_queries.reshape(batch_size, num_mix, num_frames * num_queries, 256)

        max_num_objects = 0
        for b in range(batch_size):
            for n in range(num_mix):
                probas = pred_logits[b, n].softmax(-1)[:, :-1]  # [num_frames * num_queries, num_classes]
                classes = probas.argmax(axis=-1, keepdims=True).cpu()  # [num_frames * num_queries, 1]
                scores = probas.max(axis=-1, keepdims=True).values  # [num_frames * num_queries, 1]
                frames = torch.cat([torch.full((num_queries, 1), frame_idx) for frame_idx in range(num_frames)])  # [num_frames * num_queries, 1]
                box_arrays = torch.cat([frames.cpu(), classes, scores.cpu(), pred_boxes[b, n].cpu()], dim=1).numpy()  # [num_frames * num_queries, 7]
                index = filter_detections(box_arrays, num_frames)
                if len(index) == 0 or mode != 'train':
                    index = [scores.argmax().item()]
                index_tensor = torch.LongTensor(index).to(self.opt.device)
                detections[b][n] = box_arrays[index]
                visual_features.append(torch.index_select(video_queries[b, n], 0, index_tensor))  # [num_selected_objects, 256]
                max_num_objects = max(max_num_objects, len(index))

        padded_visual_features = torch.zeros(batch_size * num_mix, max_num_objects, 256, device=self.opt.device)  # [B*N, max_num_objects, 256]
        query_key_padding_mask = torch.ones(batch_size * num_mix, max_num_objects, device=self.opt.device, dtype=torch.bool)  # [B*N, max_num_objects]
        for i, vf in enumerate(visual_features):
            padded_visual_features[i, :vf.shape[0]] = vf
            query_key_padding_mask[i, :vf.shape[0]] = False
        padded_visual_features = padded_visual_features.view(batch_size, num_mix*max_num_objects, 256)
        query_key_padding_mask = query_key_padding_mask.view(batch_size, num_mix*max_num_objects)

        # audio network forward pass
        mag_mix = input['mag_mix'] + 1e-10  # [B, 1, 512, 256]
        mags = input['mags'].view(batch_size*num_mix, 1, 512, 256)  # [B*N, 1, 512, 256]
        grid_warp = torch.from_numpy(warpgrid(mag_mix.size(0), 256, 256, warp=True)).to(self.opt.device)
        mag_mix = F.grid_sample(mag_mix, grid_warp)  # [B, 1, 256, 256]
        weight = torch.clamp(torch.log1p(mag_mix), 1e-3, 10).unsqueeze(1)  # [B, 1, 256, 256]
        grid_warp = torch.from_numpy(warpgrid(mags.size(0), 256, 256, warp=True)).to(self.opt.device)
        mags = F.grid_sample(mags, grid_warp).view(batch_size, num_mix, 1, 256, 256)  # [B, N, 1, 256, 256]

        gt_masks = mags / mag_mix.unsqueeze(1)  # [B, N, 1, 256, 256]
        gt_masks.clamp_(0., 5.)
        log_mag_mix = torch.log(mag_mix)  # [B, 1, 256, 256]

        mask_prediction = self.net_audiovisual(log_mag_mix, padded_visual_features, tgt_key_padding_mask=query_key_padding_mask)
        mask_prediction = mask_prediction.view(batch_size, num_mix, max_num_objects, 256, 256)  # [B, N, max_num_objects, 256, 256]
        
        output = {'pred_mask': mask_prediction, 'gt_mask': gt_masks, 'weight': weight, 'pred_boxes': output_pred_boxes, 'pred_logits': output_pred_logits, 'detections': detections}
        return output
    

class DETRNet(nn.Module): 
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        detr_mdl = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        detr_mdl.query_embed = nn.Embedding(opt.num_queries, opt.hidden_dim)
        detr_mdl.class_embed = nn.Linear(opt.hidden_dim, opt.num_classes + 1)
        if self.opt.pretrain_detr:
            checkpoint = torch.load(os.path.join('pretrained_models/detr', opt.detr_dataset, 'checkpoint.pth'), map_location='cpu')
            detr_mdl.load_state_dict(checkpoint['model'], strict=True)
        self.backbone = detr_mdl.backbone
        self.input_proj = detr_mdl.input_proj
        self.transformer = detr_mdl.transformer
        self.class_embed = detr_mdl.class_embed
        self.bbox_embed = detr_mdl.bbox_embed
        self.query_embed = detr_mdl.query_embed

    def forward(self, input):
        features, pos = self.backbone(input)
        src, mask = features[-1].decompose()
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'video_queries': hs[-1], 'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out


class ASTNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        ast_mdl = ASTModel(input_fdim=256, input_tdim=256, fstride=16, tstride=16, audioset_pretrain=True)
        new_proj = torch.nn.Conv2d(1, ast_mdl.original_embedding_dim, kernel_size=(16, 16), stride=(16, 16))
        new_proj.weight = torch.nn.Parameter(torch.sum(ast_mdl.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = ast_mdl.v.patch_embed.proj.bias
        ast_mdl.v.patch_embed.proj = new_proj        
        del ast_mdl.mlp_head
        self.original_dim = ast_mdl.original_embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.encoder = ast_mdl.v
        self.encoder.blocks = nn.Sequential(*list(self.encoder.blocks.children())[:6])
        self.encoder.projection_layer = nn.Linear(self.original_dim, self.hidden_dim)
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)
        self.decoder.norm = nn.LayerNorm(self.hidden_dim)
        self.unet_decoder = UNet()

    def forward(self, x, visual_features, tgt_key_padding_mask=None):
        # print(visual_features.shape) #[B, N, 256]
        # print(x.shape) #[B, 1, 256, 256]
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.encoder.patch_embed(x) #[B, 256, 768]
        x = x + self.encoder.pos_embed[:, 2:]
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        memory = self.encoder.projection_layer(x) #[B, 256(n_patch), 256(dim)]
        memory = memory.permute(1, 0, 2) #[256(n_patch), B, 256(dim)]

        query = visual_features.permute(1, 0, 2) #[N, B, 256]
        for layer in self.decoder.layers:
            output = layer(query, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.decoder.norm(output) #[N, B, 256]
        
        memory = memory.permute(1, 2, 0).reshape(B, self.hidden_dim, 16, 16) #[B, 256, H, W]
        pixel_embeddings = self.unet_decoder(memory) #[B, 256, H, W]
        output = torch.einsum("nbd,bdhw->bnhw", output, pixel_embeddings).sigmoid()
        return output
    

def unet_upconv(input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False):
    layers = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),
              norm_layer(output_nc),
              nn.ReLU(True)]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, base_channels=64, use_dropout=False):
        super(UNet, self).__init__()
        self.upconvlayer1 = unet_upconv(base_channels * 4, base_channels * 2, use_dropout=use_dropout)
        self.upconvlayer2 = unet_upconv(base_channels * 2, base_channels, use_dropout=use_dropout)
        self.upconvlayer3 = unet_upconv(base_channels, base_channels // 2, use_dropout=use_dropout)
        self.upconvlayer4 = unet_upconv(base_channels // 2, 256, use_dropout=use_dropout)  # Output has 256 channels

    def forward(self, x):
        x = self.upconvlayer1(x)
        x = self.upconvlayer2(x)
        x = self.upconvlayer3(x)
        x = self.upconvlayer4(x)
        return x  #[batch_size, 256, H, W]
    

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        output = self.norm(output)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    

def CreateModel(opt):
    net_visual = DETRNet(opt)
    net_audiovisual = ASTNet(opt)
    model = ASTDETRModel(opt, net_visual, net_audiovisual)
    if opt.weights:
        model.load_state_dict(torch.load(opt.weights), strict=True)
        print('Loaded model')
    return model
