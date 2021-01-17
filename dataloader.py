import os
import collections
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.vocab import Vocab

import numpy as np
import pandas as pd


class FeatureDataset(Dataset):

    def __init__(self, feature_path, pad_idx, max_len=50, context_len=10):
        super(FeatureDataset, self).__init__()
        self.dataset = self.load_dataset(feature_path)
        self.pad_idx = pad_idx
        self.dim_feature = self.dataset[list(self.dataset.keys())[0]].shape[1]
        self.max_len = max_len
        self.context_len = context_len

    def load_dataset(self, path) -> dict:
        raise NotImplementedError()

    def __getitem__(self, idx):
        video_id, start, end, duration = idx

        # handle data error (start and end)
        if start > end:
            start, end = end, start

        feature = np.random.randn(1, self.dim_feature) if video_id not in self.dataset else self.dataset[video_id]

        n_frames = feature.shape[0]
        start_frame = int(start / duration * n_frames)
        end_frame = int(end / duration * n_frames)
        if start_frame == end_frame:
            if start_frame == n_frames:
                start_frame -= 1
            else:
                end_frame += 1

        # video segment
        feature_frame = torch.from_numpy(feature[start_frame: end_frame, :]).float()

        # prev context_len frames as context data
        if start_frame >= self.context_len:
            feature_prev_frames = torch.from_numpy(feature[start_frame - self.context_len: start_frame, :]).float()
        elif start_frame == 0:
            feature_prev_frames = torch.randn(1, feature.shape[1]).float()
        else:
            feature_prev_frames = torch.from_numpy(feature[: start_frame, :]).float()

        # next context_len frames as context data
        if end_frame >= n_frames:
            feature_next_frames = torch.randn(1, feature.shape[1]).float()
        else:
            feature_next_frames = torch.from_numpy(feature[end_frame: end_frame + self.context_len, :]).float()

        # pad
        pad = torch.ones(max(self.max_len, self.context_len), self.dim_feature) * self.pad_idx
        feature_frame = torch.cat([feature_frame, pad])[: self.max_len, :]
        feature_prev_frames = torch.cat([feature_prev_frames, pad])[: self.context_len, :]
        feature_next_frames = torch.cat([feature_next_frames, pad])[: self.context_len, :]

        return feature_frame, feature_prev_frames, feature_next_frames

    def __len__(self):
        return len(self.dataset)


# /s1_md0/t-huluo/files/Activity_Coin_Cross

class AudioDataset(FeatureDataset):

    def __init__(self, feature_path, pad_idx, max_len=50, context_len=10):
        super(AudioDataset, self).__init__(feature_path, pad_idx, max_len, context_len)

    def load_dataset(self, path) -> dict:
        dataset = {}
        for each_file in os.listdir(path):
            audio_id = each_file[:-13] if 'features' in each_file else each_file[:-4]
            dataset[audio_id] = np.load(os.path.join(path, each_file))
        return dataset


class VideoDataset(FeatureDataset):

    def __init__(self, feature_path, pad_idx, max_len=50, context_len=10):
        super(VideoDataset, self).__init__(feature_path, pad_idx, max_len, context_len)

    def load_dataset(self, path) -> dict:
        dataset = {}
        for each_file in os.listdir(path):
            video_id = each_file[:-13] if 'features' in each_file else each_file[:-4]
            dataset[video_id] = np.load(os.path.join(path, each_file))
        return dataset


class MyDataset(Dataset):

    def __init__(self,
                 meta_path, text_path, audio_path, video_path,
                 train_set,
                 vocab_freq, modality='v',
                 init_token='<s>', eos_token='</s>',
                 pad_token='<pad>', unk_token='<unk>',
                 max_len=50, context='', on_memory=True,
                 *args, **kwargs):

        self.meta_path = meta_path
        self.text_path = text_path
        self.audio_path = audio_path
        self.video_path = video_path
        self.train_set = train_set
        self.vocab_min_freq = vocab_freq
        self.modality = modality
        self.max_len = max_len
        self.context = context
        self.on_memory = on_memory

        # text data
        self.meta_data = pd.read_csv(meta_path)

        # create vocab from data
        if not train_set:
            text_sentences = self.meta_data['text']
            caption_sentences = self.meta_data['caption']
            text_counter = collections.Counter()
            caption_counter = collections.Counter()

            for each in text_sentences:
                if not pd.isnull(each):
                    text_counter.update(each.lower().split())
            for each in caption_sentences:
                caption_counter.update(each.lower().split())
            # for each in text_counter:
            #     if text_counter[each] < vocab_freq:
            #         continue
            #     caption_counter[each] += text_counter[each]

            self.caption_vocab = Vocab(
                caption_counter, min_freq=1,
                specials=[unk_token, init_token, eos_token, pad_token]
            )
            self.text_vocab = self.caption_vocab
            #self.text_vocab = Vocab(
            #    text_counter, min_freq=vocab_freq,
            #    specials=[unk_token, init_token, eos_token, pad_token]
            #)
            self.n_src_token = len(self.text_vocab)
            self.n_tgt_token = len(self.caption_vocab)

        # validation set should inherit the vocab from train set
        else:
            self.text_vocab = train_set.text_vocab
            self.caption_vocab = train_set.caption_vocab
            self.n_src_token = train_set.n_src_token
            self.n_tgt_token = train_set.n_tgt_token

        self.init_idx = self.text_vocab.stoi[init_token]
        self.eos_idx = self.text_vocab.stoi[eos_token]
        self.pad_idx = self.text_vocab.stoi[pad_token]
        self.unk_idx = self.text_vocab.stoi[unk_token]

        self.text_dataset = json.load(open(self.text_path)) if self.text_path else {}
        self.audio_dataset = AudioDataset(audio_path, self.pad_idx, self.max_len)
        self.video_dataset = VideoDataset(video_path, self.pad_idx, self.max_len)

    def __getitem__(self, item):
        _, idx, video_id, caption, text, start, end, duration, _, _, context_text, _, *_ = self.meta_data.iloc[item]

        # text data
        # current caption
        caption_tensor = torch.tensor(self.tokenize(caption, self.caption_vocab)).long()
        # current transcript
        text_tensor = torch.tensor(self.tokenize(text, self.text_vocab)).long()
        # previous segment caption
        context_text_tensor = torch.tensor(self.tokenize(context_text, self.text_vocab)).long()

        # audio and video data
        video_tensor, prev_video_tensor, next_video_tensor = self.video_dataset[(video_id, start, end, duration)]
        audio_tensor, prev_audio_tensor, next_audio_tensor = self.audio_dataset[(video_id, start, end, duration)]

        # previous transcript
        word_sequence = self.get_word_context(video_id, start - 10000, start)
        prev_text_tensor = torch.tensor(self.tokenize(word_sequence, self.text_vocab)).long()

        # local context
        src_list = [text_tensor, audio_tensor, torch.cat([prev_video_tensor, video_tensor])]

        tgt = caption_tensor
        is_prev_tgt = torch.tensor(False).long()
        prev_tgt = torch.tensor(self.tokenize('', self.caption_vocab)).long()
                    
        if self.context == '':
            prev_context_list = [torch.zeros(0), torch.zeros(0), torch.zeros(0)]
            next_context_list = [torch.zeros(0), torch.zeros(0), torch.zeros(0)]
        elif self.context == 'frame':
            next_text_sequence = self.get_word_context(video_id, end, end + 10000)
            next_text_tensor = torch.tensor(self.tokenize(next_text_sequence, self.text_vocab)).long()
            prev_context_list = [prev_text_tensor, prev_audio_tensor, prev_video_tensor]
            next_context_list = [next_text_tensor, next_audio_tensor, next_video_tensor]
        elif 'seg' in self.context:
            if item:
                _, _, new_video_id, new_caption, new_text, new_start, new_end, new_duration, _, _, _, _, *_ = self.meta_data.iloc[item-1]
                if new_video_id != video_id:
                    context_text_tensor = torch.tensor(self.tokenize('', self.text_vocab)).long()
                    prev_audio_tensor, _, _ = self.audio_dataset[('placeholder', new_start, new_end, new_duration)]
                    prev_video_tensor, _, _ = self.video_dataset[('placeholder', new_start, new_end, new_duration)]
                else:
                    is_prev_tgt = torch.tensor(True).long()
                    prev_tgt = torch.tensor(self.tokenize(new_caption, self.caption_vocab)).long() #current caption 
                
                    if self.context == 'seg_frame':
                        context_text_tensor = torch.tensor(self.tokenize(new_text, self.text_vocab)).long()
                    prev_audio_tensor, _, _ = self.audio_dataset[(new_video_id, new_start, new_end, new_duration)]
                    prev_video_tensor, _, _ = self.video_dataset[(new_video_id, new_start, new_end, new_duration)]
            else:
                context_text_tensor = torch.tensor(self.tokenize('', self.text_vocab)).long()
                prev_audio_tensor, _, _ = self.audio_dataset[('placeholder', start, end, duration)]
                prev_video_tensor, _, _ = self.video_dataset[('placeholder', start, end, duration)]
            prev_context_list = [context_text_tensor, prev_audio_tensor, prev_video_tensor]
            #////next context
            if item < len(self.meta_data)-1:
                _, _, new_video_id, new_caption, new_text, new_start, new_end, new_duration, _, _, _, _, *_ = self.meta_data.iloc[item+1]
                if new_video_id != video_id:
                    next_context_text_tensor = torch.tensor(self.tokenize('', self.text_vocab)).long()
                    next_context_audio_tensor, _, _ = self.audio_dataset[('placeholder', new_start, new_end, new_duration)]
                    next_context_video_tensor, _, _ = self.video_dataset[('placeholder', new_start, new_end, new_duration)]
                else:                    
                    if self.context == 'seg_frame':
                        next_context_text_tensor = torch.tensor(self.tokenize(new_text, self.text_vocab)).long()
                    next_context_audio_tensor, _, _ = self.audio_dataset[(new_video_id, new_start, new_end, new_duration)]
                    next_context_video_tensor, _, _ = self.video_dataset[(new_video_id, new_start, new_end, new_duration)]
            else:
                next_context_text_tensor = torch.tensor(self.tokenize('', self.text_vocab)).long()
                next_context_audio_tensor, _, _ = self.audio_dataset[('placeholder', start, end, duration)]
                next_context_video_tensor, _, _ = self.video_dataset[('placeholder', start, end, duration)]
            next_context_list = [next_context_text_tensor, next_context_audio_tensor, next_context_video_tensor]
                
        else:
            raise ValueError(f'Unknown context type : {self.context}')

        return prev_context_list, src_list, next_context_list, tgt, prev_tgt, is_prev_tgt, (video_id, start, end)

    def __len__(self):
        return len(self.meta_data)

    def tokenize(self, sent, vocab, ignore_length=False):
        if pd.isnull(sent):
            sent = ''
        sent = sent.lower().split()
        if not ignore_length:
            sent = [vocab.stoi[each] for each in sent[:self.max_len-2]]
            pad = [self.pad_idx for _ in range(self.max_len)]
            sent = [self.init_idx] + sent + [self.eos_idx]
            return (sent + pad)[: self.max_len]
        else:
            sent = [self.init_idx] + [vocab.stoi[each] for each in sent] + [self.eos_idx]
            return sent


    def get_word_context(self, video_id, start, end):
        if video_id not in self.text_dataset:
            return ''
        sentence = []
        for word, timestamp in zip(self.text_dataset[video_id]['words'], self.text_dataset[video_id]['timestamps']):
            if start <= timestamp[0] <= end:
                sentence.append(word)
        return ' '.join(sentence)


if __name__ == '__main__':
    test_dataset = MyDataset(
        'data/train.csv', '/s1_md0/leiji/kirk_xlg/youcook/youcookii_audio_feature',
        '/s1_md0/leiji/kirk_xlg/youcook/youcookii_videos_features3dg', None, 1
    )

    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=128)
    for i, each in enumerate(test_loader):
        context, src, target, (video_id, start, end) = each

