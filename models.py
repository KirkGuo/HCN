######################################################################################
# The implementation relies on http://nlp.seas.harvard.edu/2018/04/03/attention.html #
######################################################################################

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import TokenEmbeddingLayer, FeatureEmbeddingLayer,SegmentEmbeddingLayer
from layers import PositionalEncodingLayer
from layers import BasicAttentionLayer, DoubleForwardCrossAttentionLayer
from layers import ForwardCrossAttentionLayer, BackwardCrossAttentionLayer
from layers import ConcatFusionLayer
from layers import GateSelectionLayer,GateGRUSelectionLayer, GateContextSelectionLayer


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class SelfAttentionTransformer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(SelfAttentionTransformer, self).__init__()
        self.enc_layers = clone(BasicAttentionLayer(d_model, d_model, d_model, H, d_ff, dout_p), N)

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class ForwardCrossAttentionTransformer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(ForwardCrossAttentionTransformer, self).__init__()
        self.enc_layers = clone(ForwardCrossAttentionLayer(d_model, d_model, d_model, H, d_ff, dout_p), N)

    def forward(self, x, memory_x, mask, memory_mask):
        for layer in self.enc_layers:
            x = layer(x, memory_x, mask, memory_mask)
        return x


class BackwardCrossAttentionTransformer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(BackwardCrossAttentionTransformer, self).__init__()
        self.enc_layers = clone(BackwardCrossAttentionLayer(d_model, d_model, d_model, H, d_ff, dout_p), N)

    def forward(self, x, memory_x, mask, memory_mask):
        for layer in self.enc_layers:
            x = layer(x, memory_x, mask, memory_mask)
        return x


class DoubleForwardCrossAttentionTransformer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(DoubleForwardCrossAttentionTransformer, self).__init__()
        self.dec_layers = clone(DoubleForwardCrossAttentionLayer(d_model, d_model, d_model, H, d_ff, dout_p), N)

    def forward(self, x, context_x, encoder_x, mask, context_mask, encoder_mask):
        for layer in self.dec_layers:
            x = layer(x, context_x, encoder_x, mask, context_mask, encoder_mask)
        return x


class SimpleEncoderDecoderCat(nn.Module):

    def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
        super(SimpleEncoderDecoderCat, self).__init__()
        self.modality = modality

        if 'v' in self.modality:
            self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
            self.tgt_emb_video = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
            self.pos_emb_video = PositionalEncodingLayer(dim_model, prob_dropout)
            self.encoder_video = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
            self.decoder_video = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        if 't' in self.modality:
            self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
            self.tgt_emb_text = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
            self.pos_emb_text = PositionalEncodingLayer(dim_model, prob_dropout)
            self.encoder_text = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
            self.decoder_text = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        if 'a' in self.modality:
            self.src_emb_audio = FeatureEmbeddingLayer(dim_feature, dim_model)
            self.tgt_emb_audio = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
            self.pos_emb_audio = PositionalEncodingLayer(dim_model, prob_dropout)
            self.encoder_audio = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
            self.decoder_audio = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        
        self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)

        self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
        self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)

        dim_cat_feature = 0
        if 'v' in self.modality:
            dim_cat_feature += dim_model
        if 'a' in self.modality:
            dim_cat_feature += dim_model
        if 't' in self.modality:
            dim_cat_feature += dim_model

        self.generator = ConcatFusionLayer(dim_cat_feature, n_tgt_vocab, prob_dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, contexts, sources, next_contexts, target, masks, iscontext):
        context_masks, src_masks, next_context_masks, target_mask = masks
        context_text, context_audio, context_video = contexts
        text, audio, video = sources
        context_text_mask, context_audio_mask, context_video_mask = context_masks
        text_mask, audio_mask, video_mask = src_masks	
        next_context_text, next_context_audio, next_context_video = next_contexts
        next_context_text_mask, next_context_audio_mask, next_context_video_mask = next_context_masks

    #def forward(self, contexts, srcs, target, masks):
    #    context_masks, src_masks, target_mask = masks
    #    text, audio, video = srcs
    #    text_mask, audio_mask, video_mask = src_masks
        out_feature = []
        if 'v' in self.modality:
            out_video = self.src_emb_video(video)
            out_video = self.pos_emb_video(out_video)
            out_video = self.encoder_video(out_video, video_mask)
            #video_target = self.tgt_emb_video(target)
            #video_target = self.tgt_pos_emb(video_target)
            #out_video = self.decoder_video(video_target, out_video, target_mask, video_mask)
            out_feature.append(out_video)
        if 't' in self.modality:
            out_text = self.src_emb_text(text)
            out_text = self.pos_emb_text(out_text)
            out_text = self.encoder_text(out_text, text_mask)
            #text_target = self.tgt_emb_text(target)
            #text_target = self.tgt_pos_emb(text_target)
            #out_text = self.decoder_text(text_target, out_text, target_mask, text_mask)
            out_feature.append(out_text)
        if 'a' in self.modality:
            out_audio = self.src_emb_audio(audio)
            out_audio = self.pos_emb_audio(out_audio)
            out_audio = self.encoder_audio(out_audio, audio_mask)
            #audio_target = self.tgt_emb_audio(target)
            #audio_target = self.tgt_pos_emb(audio_target)
            #out_audio = self.decoder_audio(audio_target, out_audio, target_mask, audio_mask)
            out_feature.append(out_audio)

        out = torch.cat(out_feature, dim=-1)
        #out = torch.cat((out_video, out_text), dim=-1)
        out_mask = torch.cat((video_mask, text_mask), dim=-1)
        target = self.tgt_emb(target)
        target = self.tgt_pos_emb(target)
        out = self.decoder(target, out, target_mask, out_mask)
            
        out = self.generator(out)
        return out

# class SimpleGateEncoderDecoderCat(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(SimpleGateEncoderDecoderCat, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#             self.tgt_emb_video = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#             self.pos_emb_video = PositionalEncodingLayer(dim_model, prob_dropout)
#             self.encoder_video = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#             self.decoder_video = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#             self.cxt_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#             self.context_encoder_video = BackwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#             self.encoder_gate_video = GateSelectionLayer(dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#             self.tgt_emb_text = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#             self.pos_emb_text = PositionalEncodingLayer(dim_model, prob_dropout)
#             self.encoder_text = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#             self.decoder_text = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#             self.cxt_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#             self.context_encoder_text = BackwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#             self.encoder_gate_text = GateSelectionLayer(dim_model)
#
#
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         dim_cat_feature = 0
#         if 'v' in self.modality:
#             dim_cat_feature += dim_model
#         if 'a' in self.modality:
#             dim_cat_feature += dim_model
#         if 't' in self.modality:
#             dim_cat_feature += dim_model
#
#         self.generator = ConcatFusionLayer(dim_cat_feature, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, contexts, srcs, target, masks):
#         context_masks, src_masks, target_mask = masks
#         context_text, context_audio, context_video = contexts
#         text, audio, video = srcs
#         context_text_mask, context_audio_mask, context_video_mask = context_masks
#         text_mask, audio_mask, video_mask = src_masks
#
#         out_feature = []
#         if 'v' in self.modality:
#             out_video = self.src_emb_video(video)
#             out_video = self.pos_emb_video(out_video)
#
#             out_context = self.cxt_emb_video(context_video)
#             out_context = self.pos_emb_video(out_context)
#
#             out_video = self.encoder_video(out_video, video_mask)
#             out_encoder_context = self.context_encoder_video(out_context, out_video, context_video_mask, None)
#             out_encoder = self.encoder_gate_video(out_encoder_context, out_video)
#
#
#             video_target = self.tgt_emb_video(target)
#             video_target = self.tgt_pos_emb(video_target)
#             out_video = self.decoder_video(video_target, out_encoder, target_mask, video_mask)
#             out_feature.append(out_video)
#
#
#         if 't' in self.modality:
#
#             out_text = self.src_emb_text(text)
#             out_text = self.pos_emb_text(out_text)
#
#             out_context = self.cxt_emb_text(context_text)
#             out_context = self.pos_emb_text(out_context)
#
#             out_text = self.encoder_text(out_text, text_mask)
#             out_encoder_context = self.context_encoder_text(out_context, out_text, context_text_mask, None)
#             out_encoder = self.encoder_gate_text(out_encoder_context, out_text)
#
#
#             text_target = self.tgt_emb_text(target)
#             text_target = self.tgt_pos_emb(text_target)
#             out_text = self.decoder_text(text_target, out_encoder, target_mask, text_mask)
#             out_feature.append(out_text)
#
#
#         out_feature = torch.cat(out_feature, dim=-1)
#         out = self.generator(out_feature)
#         return out


# class MMSimpleEncoderDecoderCat(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(MMSimpleEncoderDecoderCat, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#         if 'a' in self.modality:
#             self.src_emb_audio = FeatureEmbeddingLayer(dim_feature, dim_model)
#
#         self.pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#         self.encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         #self.type_emb = nn.Embedding(config.type_vocab_size, config.hidden_size)
#
#         #dim_cat_feature = 0
#         #dim_cat_feature += dim_model
#
#         self.generator = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, contexts, srcs, target, masks):
#         context_masks, src_masks, target_mask = masks
#         text, audio, video = srcs
#         text_mask, audio_mask, video_mask = src_masks
#         out_feature = []
#         out_mask = []
#
#
#         #token_type_embeddings = self.token_type_embeddings(concat_type)
#         #position_embeddings = self.position_embeddings(position_ids)
#         #embeddings = concat_embeddings + position_embeddings + token_type_embeddings
#
#
#         out_video = self.src_emb_video(video)
#         out_video = self.pos_emb(out_video)
#         out_text = self.src_emb_text(text)
#         out_text = self.pos_emb(out_text)
#
#         out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
#         out_mask = torch.cat((video_mask, text_mask), dim=-1)
#
#         out_feature = self.encoder(out_feature, out_mask)
#         target_feature = self.tgt_emb(target)
#         target_feature = self.tgt_pos_emb(target_feature)
#         out_feature = self.decoder(target_feature, out_feature, target_mask, out_mask)
#
#
#         out = self.generator(out_feature)
#         return out


# class UnifiedSimpleEncoderDecoderCat(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(UnifiedSimpleEncoderDecoderCat, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#         if 'a' in self.modality:
#             self.src_emb_audio = FeatureEmbeddingLayer(dim_feature, dim_model)
#
#         self.pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#         self.encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         #self.type_emb = nn.Embedding(config.type_vocab_size, config.hidden_size)
#
#         #dim_cat_feature = 0
#         #dim_cat_feature += dim_model
#
#         self.generator = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, contexts, srcs, target, masks):
#         context_masks, src_masks, target_mask = masks
#         text, audio, video = srcs
#         text_mask, audio_mask, video_mask = src_masks
#         out_feature = []
#         out_mask = []
#
#         # 1. type
#         if 'v' in self.modality:
#             out_video = self.src_emb_video(video)
#             out_video = self.pos_emb(out_video)
#             out_feature = out_video
#             out_mask = video_mask
#         if 't' in self.modality:
#             out_text = self.src_emb_text(text)
#             out_text = self.pos_emb(out_text)
#             out_feature = out_text
#             out_mask = text_mask
#         if 'a' in self.modality:
#             out_audio = self.src_emb_audio(audio)
#             out_audio = self.pos_emb(out_audio)
#             out_feature = out_audio
#             out_mask = audio_mask
#
#
#         if 'v' in self.modality and 't' in self.modality:
#             out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask = torch.cat((video_mask, text_mask), dim=-1)
#         if 'v' in self.modality and 'a' in self.modality:
#             out_feature = torch.cat((out_video, out_audio), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask = torch.cat((video_mask, audio_mask), dim=-1)
#
#         out_feature = self.encoder(out_feature, out_mask)
#         target_feature = self.tgt_emb(target)
#         target_feature = self.tgt_pos_emb(target_feature)
#         out_feature = self.decoder(target_feature, out_feature, target_mask, out_mask)
#
#         out = self.generator(out_feature)
#         return out

# class UnifiedGateEncoderDecoderCat(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(UnifiedGateEncoderDecoderCat, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#
#         self.src_type_emb = SegmentEmbeddingLayer(2, 3, dim_model)
#         self.src_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         #self.multimodal_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.cross_encoder = BackwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         #self.encoder_gate = GateSelectionLayer(dim_model, dim_ff, prob_dropout)
#         self.encoder_gate = GateGRUSelectionLayer(dim_model, dim_ff, prob_dropout)
#
#         self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         self.generator = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, contexts, srcs, target, masks, iscontext):
#         context_masks, src_masks, target_mask = masks
#         context_text, context_audio, context_video = contexts
#         text, audio, video = srcs
#         context_text_mask, context_audio_mask, context_video_mask = context_masks
#         text_mask, audio_mask, video_mask = src_masks
#
#         out_feature = []
#         out_mask = []
#         out_feature_context = []
#         out_mask_context = []
#
#         if 'v' in self.modality:
#             out_video = self.src_emb_video(video)
#             out_video = self.src_pos_emb(out_video)
#             out_video = self.src_type_emb(out_video, 0, 0)
#
#             out_feature = out_video
#             out_mask = video_mask
#
#             if iscontext != 0:
#                 out_video_context = self.src_emb_video(context_video)
#                 out_video_context = self.src_pos_emb(out_video_context)
#                 out_video_context = self.src_type_emb(out_video_context, 1, iscontext)
#                 out_feature_context = out_video_context
#                 out_mask_context = context_video_mask
#         if 't' in self.modality:
#             out_text = self.src_emb_text(text)
#             out_text = self.src_pos_emb(out_text)
#             out_text = self.src_type_emb(out_text, 0, 0)
#
#             out_feature = out_text
#             out_mask = text_mask
#
#             if iscontext != 0:
#                 out_text_context = self.src_emb_text(context_text)
#                 out_text_context = self.src_pos_emb(out_text_context)
#                 out_text_context = self.src_type_emb(out_text_context, 1, iscontext)
#                 out_feature_context = out_text_context
#                 out_mask_context = context_text_mask
#
#         if 'v' in self.modality and 't' in self.modality:
#             out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask = torch.cat((video_mask, text_mask), dim=-1)
#             #out_feature = self.multimodal_encoder(out_feature, out_mask)
#
#             if iscontext != 0:
#                 out_feature_context = torch.cat((out_video_context, out_text_context), dim=1) #torch.cat(out_feature, dim=0)
#                 out_mask_context = torch.cat((context_video_mask, context_text_mask), dim=-1)
#                 #out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)
#
#         out_encoder = self.encoder(out_feature, out_mask)
#         if iscontext != 0:
#             out_feature_context = self.encoder(out_feature_context, out_mask_context)
#             out_encoder_context = self.cross_encoder(out_feature_context, out_encoder, out_mask_context, None)
#             out_encoder = self.encoder_gate(out_encoder_context, out_encoder)
#
#
#         target = self.tgt_emb(target)
#         target = self.tgt_pos_emb(target)
#         out_decoder = self.decoder(target, out_encoder, target_mask, out_mask)
#
#         out = self.generator(out_decoder)
#         return out

# class UnifiedUniEncoderTransformer(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(UnifiedUniEncoderTransformer, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#             #self.cxt_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#             #self.cxt_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#
#         self.src_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#         #self.cxt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         self.multimodal_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.bottom_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         self.top_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#
#         self.to_caption = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#
#     def forward(self, contexts, srcs, target, masks):
#         context_masks, src_masks, target_mask = masks
#         context_text, context_audio, context_video = contexts
#         text, audio, video = srcs
#         context_text_mask, context_audio_mask, context_video_mask = context_masks
#         text_mask, audio_mask, video_mask = src_masks
#
#         out_feature = []
#         out_mask = []
#         out_feature_context = []
#         out_mask_context = []
#
#         if 'v' in self.modality:
#             out_video = self.src_emb_video(video)
#             out_video_context = self.src_emb_video(context_video)
#
#             out_video = self.src_pos_emb(out_video,0)
#             out_video_context = self.src_pos_emb(out_video_context,1)
#             #out_video = self.src_pos_emb(out_video)
#             #out_video_context = self.src_pos_emb(out_video_context)
#
#             out_feature = out_video
#             out_mask = video_mask
#             out_feature_context = out_video_context
#             out_mask_context = context_video_mask
#
#         if 't' in self.modality:
#             out_text = self.src_emb_text(text)
#             out_text_context = self.src_emb_text(context_text)
#
#             out_text = self.src_pos_emb(out_text,0)
#             out_text_context = self.src_pos_emb(out_text_context,1)
#
#             out_feature = out_text
#             out_mask = text_mask
#             out_feature_context = out_text_context
#             out_mask_context = context_text_mask
#
#         if 'v' in self.modality and 't' in self.modality:
#             out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask = torch.cat((video_mask, text_mask), dim=-1)
#
#             out_feature_context = torch.cat((out_video_context, out_text_context), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask_context = torch.cat((context_video_mask, context_text_mask), dim=-1)
#             # 4. cross / type
#             out_feature = self.multimodal_encoder(out_feature, out_mask)
#             out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)
#
#
#         out = torch.cat([out_feature_context, out_feature], dim=1)
#         out_mask = torch.cat([out_mask_context, out_mask], dim=-1)
#         out = self.bottom_encoder(out, out_mask)
#
#         out = out[:, out_feature_context.size(1):, :]
#         # 3. top
#         out = self.top_encoder(out, None)
#
#         out_target = self.tgt_emb(target)
#         out_target = self.tgt_pos_emb(out_target)
#         out = self.decoder(out_target, out, target_mask, None)
#
#         out = self.to_caption(out)
#         return out

# iscontext=0: only sourec
# iscontext=1: context+source
# iscontext=2: source+context
# typeid: 0 for s , 1 for c
# segmentid: 0 for s, 1 for p,2 for n

# class UnifiedBiEncoderTransformer(nn.Module):
#
#     def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
#         super(UnifiedBiEncoderTransformer, self).__init__()
#         self.modality = modality
#
#         if 'v' in self.modality:
#             self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)
#         if 't' in self.modality:
#             self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
#
#         self.src_type_emb = SegmentEmbeddingLayer(2, 3, dim_model)
#         self.src_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         self.multimodal_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.bottom_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#         self.top_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
#
#         self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
#         self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)
#
#         self.to_caption = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, contexts, srcs, next_contexts, target, masks, iscontext):
#         context_masks, src_masks, next_context_masks, target_mask = masks
#         context_text, context_audio, context_video = contexts
#         text, audio, video = srcs
#         context_text_mask, context_audio_mask, context_video_mask = context_masks
#         text_mask, audio_mask, video_mask = src_masks
#
#         if iscontext == 2:
#             context_text, context_audio, context_video = next_contexts
#             context_text_mask, context_audio_mask, context_video_mask = next_context_masks
#
#         out_feature = []
#         out_mask = []
#         out_feature_context = []
#         out_mask_context = []
#
#         if 'v' in self.modality:
#             out_video = self.src_emb_video(video)
#             out_video = self.src_pos_emb(out_video)
#             out_video = self.src_type_emb(out_video, 0, 0)
#
#             out_feature = out_video
#             out_mask = video_mask
#
#             if iscontext != 0:
#                 out_video_context = self.src_emb_video(context_video)
#                 out_video_context = self.src_pos_emb(out_video_context)
#                 out_video_context = self.src_type_emb(out_video_context, 1, iscontext)
#                 out_feature_context = out_video_context
#                 out_mask_context = context_video_mask
#         if 't' in self.modality:
#             out_text = self.src_emb_text(text)
#             out_text = self.src_pos_emb(out_text)
#             out_text = self.src_type_emb(out_text, 0, 0)
#
#             out_feature = out_text
#             out_mask = text_mask
#
#             if iscontext != 0:
#                 out_text_context = self.src_emb_text(context_text)
#                 out_text_context = self.src_pos_emb(out_text_context)
#                 out_text_context = self.src_type_emb(out_text_context, 1, iscontext)
#                 out_feature_context = out_text_context
#                 out_mask_context = context_text_mask
#
#         if 'v' in self.modality and 't' in self.modality:
#             out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
#             out_mask = torch.cat((video_mask, text_mask), dim=-1)
#             #out_feature = self.multimodal_encoder(out_feature, out_mask)
#
#             if iscontext != 0:
#                 out_feature_context = torch.cat((out_video_context, out_text_context), dim=1) #torch.cat(out_feature, dim=0)
#                 out_mask_context = torch.cat((context_video_mask, context_text_mask), dim=-1)
#                 #out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)
#
#
#         if iscontext != 0:
#             out = torch.cat([out_feature_context, out_feature], dim=1)
#             out_mask = torch.cat([out_mask_context, out_mask], dim=-1)
#         else:
#             out = out_feature
#             out_mask = out_mask
#
#         out = self.bottom_encoder(out, out_mask)
#
#         if iscontext != 0:
#             out = out[:, out_feature_context.size(1):, :]
#
#         # leiji: whether top layer
#         out = self.top_encoder(out, None)
#
#         out_target = self.tgt_emb(target)
#         out_target = self.tgt_pos_emb(out_target)
#         out = self.decoder(out_target, out, target_mask, None)
#
#         out = self.to_caption(out)
#         return out


class UnifiedTriEncoderTransformer(nn.Module):
        
    def __init__(self, n_tgt_vocab, dim_feature, dim_model, dim_ff, H, N, prob_dropout, modality, *args, **kwargs):
        super(UnifiedTriEncoderTransformer, self).__init__()
        self.modality = modality

        if 'v' in self.modality:
            self.src_emb_video = FeatureEmbeddingLayer(dim_feature, dim_model)  
        if 't' in self.modality:
            self.src_emb_text = TokenEmbeddingLayer(kwargs['n_src_vocab'], dim_model)
        if 'a' in self.modality:
            self.src_emb_audio = FeatureEmbeddingLayer(dim_model, dim_model)  

        self.src_type_emb = SegmentEmbeddingLayer(2, 3, dim_model)
        self.src_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)

        self.multimodal_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        #####flat
        #self.bottom_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        #self.top_encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)

        #####gate
        self.encoder = SelfAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        self.cross_encoder = BackwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)
        #self.internal_gate = GateContextSelectionLayer(dim_model, dim_ff, prob_dropout)
        self.internal_gate = GateGRUSelectionLayer(dim_model, dim_ff, prob_dropout)
        self.encoder_gate = GateContextSelectionLayer(dim_model, dim_ff, prob_dropout)
        self.context_gate = GateContextSelectionLayer(dim_model, dim_ff, prob_dropout)

        #self.encoder_gate = GateGRUSelectionLayer(dim_model, dim_ff, prob_dropout)
        #self.context_gate = GateGRUSelectionLayer(dim_model, dim_ff, prob_dropout)

        self.decoder = ForwardCrossAttentionTransformer(dim_model, prob_dropout, H, dim_ff, N)

        self.tgt_emb = TokenEmbeddingLayer(n_tgt_vocab, dim_model)
        self.tgt_pos_emb = PositionalEncodingLayer(dim_model, prob_dropout)

        self.to_caption = ConcatFusionLayer(dim_model, n_tgt_vocab, prob_dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def onepass(self, video, video_mask, text, text_mask, audio, audio_mask, iscontext, context_video, context_video_mask, context_text, context_text_mask, context_audio, context_audio_mask):
        out_feature = []
        out_mask = []
        out_feature_context = []
        out_mask_context = []

        if 'v' in self.modality:
            out_video = self.src_emb_video(video)
            out_video = self.src_pos_emb(out_video)
            out_video = self.src_type_emb(out_video, 0, 0)

            out_feature = out_video
            out_mask = video_mask           

            if iscontext != 0:
                out_video_context = self.src_emb_video(context_video)
                out_video_context = self.src_pos_emb(out_video_context)
                out_video_context = self.src_type_emb(out_video_context, 1, iscontext)
                out_feature_context = out_video_context
                out_mask_context = context_video_mask
        if 't' in self.modality:
            out_text = self.src_emb_text(text)
            out_text = self.src_pos_emb(out_text)
            out_text = self.src_type_emb(out_text, 0, 0)

            out_feature = out_text
            out_mask = text_mask           

            # if False: #iscontext != 0:
            #     out_text_context = self.src_emb_text(context_text)
            #     out_text_context = self.src_pos_emb(out_text_context)
            #     out_text_context = self.src_type_emb(out_text_context, 1, iscontext)
            #     out_feature_context = out_text_context
            #     out_mask_context = context_text_mask
        if 'a' in self.modality:
            out_audio = self.src_emb_audio(audio)
            out_audio= self.src_pos_emb(out_audio)
            out_audio= self.src_type_emb(out_audio, 0, 0)

            out_feature = out_audio
            out_mask = audio_mask           

            if iscontext != 0:
                out_audio_context = self.src_emb_audio(context_audio)
                out_audio_context = self.src_pos_emb(out_audio_context)
                out_audio_context = self.src_type_emb(out_audio_context, 1, iscontext)
                out_feature_context = out_audio_context
                out_mask_context = context_audio_mask

        if 'v' in self.modality and 't' in self.modality:
            out_feature = torch.cat((out_video, out_text), dim=1) #torch.cat(out_feature, dim=0)
            out_mask = torch.cat((video_mask, text_mask), dim=-1)
            
            # if False: #iscontext != 0:
            #     out_feature_context = torch.cat((out_video_context, out_text_context), dim=1) #torch.cat(out_feature, dim=0)
            #     out_mask_context = torch.cat((context_video_mask, context_text_mask), dim=-1)
            #     #out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)

        if 'v' in self.modality and 'a' in self.modality:
            if 't' not in self.modality:
                out_feature = out_video
                out_mask = video_mask 
                if iscontext != 0:          
                    out_feature_context = out_video_context
                    out_mask_context = context_video_mask
                
            out_feature = torch.cat((out_feature, out_audio), dim=1)
            out_mask = torch.cat((out_mask, audio_mask), dim=-1)
                
            if iscontext != 0:
                out_feature_context = torch.cat((out_feature_context, out_audio_context), dim=1) 
                out_mask_context = torch.cat((out_mask_context, context_audio_mask), dim=-1)
                #out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)     

        out = out_feature
        out_mask = out_mask

        ####flat
        #if iscontext != 0:
        #    out = torch.cat([out_feature_context, out_feature], dim=1)
        #    out_mask = torch.cat([out_mask_context, out_mask], dim=-1)
        #out = self.bottom_encoder(out, out_mask)
        #if iscontext != 0:
        #    out = out[:, out_feature_context.size(1):, :]
        # leiji: whether top layer
        #out = self.top_encoder(out, None)
        
        ####gate
        out = self.encoder(out, out_mask)
        ##### if local context
        context_len = 10
        out = out[:, context_len:, :]
        out_mask = out_mask[:, context_len:, :] 
        if iscontext != 0:
            out_feature_context = self.encoder(out_feature_context, out_mask_context)
            out_encoder_context = self.cross_encoder(out_feature_context, out, out_mask_context, None)
            out = self.internal_gate(out_encoder_context, out)

        return out

    def new_onepass(self, context, src, context_mask, src_mask, is_context):
        out_feature = []
        out_mask = []
        out_feature_context = []
        out_mask_context = []

        if 'v' in self.modality:
            out_video = self.src_emb_video(src[2])
            out_video = self.src_pos_emb(out_video)
            out_video = self.src_type_emb(out_video, 0, 0)

            out_feature.append(out_video)
            out_mask.append(src_mask[2])

            if is_context:
                out_video_context = self.src_emb_video(context[2])
                out_video_context = self.src_pos_emb(out_video_context)
                out_video_context = self.src_type_emb(out_video_context, 1, is_context)
                out_feature_context.append(out_video_context)
                out_mask_context.append(context_mask[2])

        if 't' in self.modality:
            out_text = self.src_emb_text(src[0])
            out_text = self.src_pos_emb(out_text)
            out_text = self.src_type_emb(out_text, 0, 0)

            out_feature.append(out_text)
            out_mask.append(src_mask[0])

            # if False: #iscontext != 0:
            #     out_text_context = self.src_emb_text(context_text)
            #     out_text_context = self.src_pos_emb(out_text_context)
            #     out_text_context = self.src_type_emb(out_text_context, 1, iscontext)
            #     out_feature_context = out_text_context
            #     out_mask_context = context_text_mask

        if 'a' in self.modality:
            out_audio = self.src_emb_audio(src[1])
            out_audio = self.src_pos_emb(out_audio)
            out_audio = self.src_type_emb(out_audio, 0, 0)

            out_feature.append(out_audio)
            out_mask.append(src_mask[1])

            if is_context:
                out_audio_context = self.src_emb_audio(context[1])
                out_audio_context = self.src_pos_emb(out_audio_context)
                out_audio_context = self.src_type_emb(out_audio_context, 1, is_context)
                out_feature_context.append(out_audio_context)
                out_mask_context.append(context_mask[1])

        # if 'v' in self.modality and 't' in self.modality:
        #     out_feature = torch.cat(out_feature, dim=1)  # torch.cat(out_feature, dim=0)
        #     out_mask = torch.cat((video_mask, text_mask), dim=-1)
        #
        #     # if False: #iscontext != 0:
        #     #     out_feature_context = torch.cat((out_video_context, out_text_context), dim=1) #torch.cat(out_feature, dim=0)
        #     #     out_mask_context = torch.cat((context_video_mask, context_text_mask), dim=-1)
        #     #     #out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)

        # if 'v' in self.modality and 'a' in self.modality:
        #     if 't' not in self.modality:
        #         out_feature = out_video
        #         out_mask = video_mask
        #         if iscontext != 0:
        #             out_feature_context = out_video_context
        #             out_mask_context = context_video_mask
        #
        #     out_feature = torch.cat((out_feature, out_audio), dim=1)
        #     out_mask = torch.cat((out_mask, audio_mask), dim=-1)
        #
        #     if iscontext != 0:
        #         out_feature_context = torch.cat((out_feature_context, out_audio_context), dim=1)
        #         out_mask_context = torch.cat((out_mask_context, context_audio_mask), dim=-1)
        #         # out_feature_context = self.multimodal_encoder(out_feature_context, out_mask_context)

        out = torch.cat(out_feature, dim=1)
        out_mask = torch.cat(out_mask, dim=-1)

        if is_context:
            out_feature_context = torch.cat(out_feature_context, dim=1)
            out_mask_context = torch.cat(out_mask_context, dim=-1)
            # print(out_feature_context.size(), out_mask_context.size(), is_context)


        ####flat
        # if iscontext != 0:
        #    out = torch.cat([out_feature_context, out_feature], dim=1)
        #    out_mask = torch.cat([out_mask_context, out_mask], dim=-1)
        # out = self.bottom_encoder(out, out_mask)
        # if iscontext != 0:
        #    out = out[:, out_feature_context.size(1):, :]
        # leiji: whether top layer
        # out = self.top_encoder(out, None)

        ####gate
        out = self.encoder(out, out_mask)
        ##### if local context
        context_len = 10
        out = out[:, context_len:, :]
        # out_mask = out_mask[:, context_len:, :]
        if is_context:
            out_feature_context = self.encoder(out_feature_context, out_mask_context)
            out_encoder_context = self.cross_encoder(out_feature_context, out, out_mask_context, None)
            out = self.internal_gate(out_encoder_context, out)

        return out

    #iscontext=0: only sourec
    #iscontext=1: context+source
    #iscontext=2: source+context
    #typeid: 0 for s , 1 for c
    #segmentid: 0 for s, 1 for p,2 for n

    def forward(self, contexts, sources, next_contexts, target, masks, iscontext):

        context_masks, src_masks, next_context_masks, target_mask = masks

        context_text, context_audio, context_video = contexts
        text, audio, video = sources
        next_context_text, next_context_audio, next_context_video = next_contexts

        context_text_mask, context_audio_mask, context_video_mask = context_masks
        text_mask, audio_mask, video_mask = src_masks
        next_context_text_mask, next_context_audio_mask, next_context_video_mask = next_context_masks

        v_1 = context_video
        v_2 = video
        v_3 = next_context_video
        t_1 = context_text
        t_2 = text
        t_3 = next_context_text
        a_1 = context_audio
        a_2 = audio
        a_3 = next_context_audio

        vmask_1 = context_video_mask
        vmask_2 = video_mask
        vmask_3 = next_context_video_mask
        tmask_1 = context_text_mask
        tmask_2 = text_mask
        tmask_3 = next_context_text_mask
        amask_1 = context_audio_mask
        amask_2 = audio_mask
        amask_3 = next_context_audio_mask

        out = None
        
        # 0 is s
        # 1 as p,s, 2 as s,n
        # 3 as p,s,n
        # if iscontext == 0:
        #     out = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, iscontext, None, None, None, None, None, None)
        # elif iscontext == 1:
        #     out = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, iscontext, v_1, vmask_1, t_1, tmask_1, a_1, amask_1)
        # elif iscontext == 2:
        #     out = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, iscontext, v_3, vmask_3, t_3, tmask_3,a_3, amask_3)
        # elif iscontext == 3:
        #     out_P = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 1, v_1, vmask_1, t_1, tmask_1, a_1, amask_1)
        #     out_n = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 2, v_3, vmask_3, t_3, tmask_3,a_3, amask_3)
        #     out = self.context_gate(out_P, out_n)
        #     # gate option
        #     out_s = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 0, None, None, None, None, None, None)
        #     out = self.encoder_gate(out, out_s)

        # out_P = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 1, v_1, vmask_1, t_1, tmask_1, a_1, amask_1)
        # out_n = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 2, v_3, vmask_3, t_3, tmask_3, a_3, amask_3)
        # out = self.context_gate(out_P, out_n)
        # out_s = self.onepass(v_2, vmask_2, t_2, tmask_2, a_2, amask_2, 0, None, None, None, None, None, None)
        # out = self.encoder_gate(out, out_s)

        if iscontext == 0:
            out = self.new_onepass(None, sources, None, src_masks, 0)
        elif iscontext == 1:
            out = self.new_onepass(contexts, sources, context_masks, src_masks, 1)
        elif iscontext == 2:
            out = self.new_onepass(next_contexts, sources, next_context_masks, src_masks, 2)
        else:
            out_P = self.new_onepass(contexts, sources, context_masks, src_masks, 1)
            out_n = self.new_onepass(next_contexts, sources, next_context_masks, src_masks, 2)
            out = self.context_gate(out_P, out_n)
            out_s = self.new_onepass(None, sources, None, src_masks, 0)
            out = self.encoder_gate(out, out_s)

        out_target = self.tgt_emb(target)
        out_target = self.tgt_pos_emb(out_target)
        out = self.decoder(out_target, out, target_mask, None)

        out = self.to_caption(out)
        return out
