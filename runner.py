import torch
import utils
import pickle
import json
import os
import re
import sys
import io

from evaluate import ANETcaptions
from nlgeval import compute_metrics, NLGEval


def __calculate_scores(result_file, ref_file, block_print=True):
    reference_file = json.load(open(ref_file))
    ref_video_keys = sorted(list(reference_file.keys()))
    ref_text_list = sum([reference_file[item]['sentences'] for item in ref_video_keys], [])

    file_data = json.load(open(result_file))
    hyp_text_list = sum([[i['sentence'].lower() for i in file_data['results'][item]] for item in ref_video_keys], [])
    hyp_text_list = ['<NONE>' if len(item) == 0 else item for item in hyp_text_list]  # for empty generated result
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

    result = nlgeval.compute_metrics(hyp_list=ref_text_list, ref_list=[hyp_text_list])
    metrics = {
        'Average across tIoUs': result
    }
    return metrics


class EpochRunner:

    def __init__(self, model, train_loader, test_loader, scheduler, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.init_idx = self.train_loader.dataset.init_idx
        self.pad_idx = self.train_loader.dataset.pad_idx
        self.eos_idx = self.train_loader.dataset.eos_idx
        self.args = args

    def train(self, epoch):
        self.model.train()
        
        total_loss = 0.0
        count = 0
        for i, (context, src, next_context, target, prev_tgt, is_prev_tgt, *_) in enumerate(self.train_loader):
            context, src, next_context, target,  masks = self.__data_to_model(context, src, next_context, target)
            n_token = (target[1] != self.pad_idx).sum()

            if epoch <= self.args.warm_up:
                output = self.model(context, src, next_context, target[0], masks, 0)
                loss = self.scheduler(output, target[1], n_token)
                self.scheduler.step(epoch)
                total_loss += (loss / n_token).item()
                output = self.model(context, src, next_context, target[0], masks, 1)
                loss = self.scheduler(output, target[1], n_token)
                self.scheduler.step(epoch)
                total_loss += (loss / n_token).item()
                output = self.model(context, src, next_context, target[0], masks, 2)
                loss = self.scheduler(output, target[1], n_token)
                self.scheduler.step(epoch)
                total_loss += (loss / n_token).item()

            # add self
            # if 's' in self.args.data:
            #     if ('a' not in self.args.data and 'p' not in self.args.data and 'n' not in self.args.data) or epoch < self.args.warm_up:
            #         output = self.model(context, src, next_context, target[0], masks, 0)
            #         loss = self.scheduler(output, target[1], n_token)
            #         if torch.isnan(loss):
            #             print('debug here')
            #         self.scheduler.step(epoch)
            #
            #         total_loss += (loss / n_token).item()
            #         count = count + 1

            # add prev
            # if 'p' in self.args.data:
            #     if 'n' not in self.args.data or epoch <=  self.args.warm_up:
            #         output = self.model(context, src, next_context, target[0], masks, 1)
            #         loss = self.scheduler(output, target[1], n_token)
            #         if torch.isnan(loss):
            #             print('debug here')
            #         self.scheduler.step(epoch)
            #
            #         total_loss += (loss / n_token).item()
            #         count = count+1

            # # add next, not the final last one
            # if 'n' in self.args.data:           # and is_prev_tgt[0] == True:
            #     if 'p' not in self.args.data or epoch <= self.args.warm_up:
            #         output = self.model(context, src,  next_context, target[0], masks , 2)
            #         loss = self.scheduler(output, target[1], n_token)
            #         if torch.isnan(loss):
            #             print('debug here')
            #         self.scheduler.step(epoch)
            #
            #         total_loss += (loss / n_token).item()
            #         count=count+1

            # if 'spn' in self.args.data or 'a' in self.args.data:
            #     if True:  #epoch >= self.args.warm_up or 'a' in self.args.data:
            output = self.model(context, src, next_context, target[0], masks, 3)
            loss = self.scheduler(output, target[1], n_token)
            self.scheduler.step(epoch)
            total_loss += (loss / n_token).item()

        total_loss_norm = total_loss / len(self.train_loader)
        print(f'gpu({self.args.gpu}),epoch({epoch}): train loss: {total_loss_norm}:count:{count}') if self.args.verbose else False
        return total_loss_norm

    def eval(self, epoch, min_loss):
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            # compute validation loss
            for i, (context, src, next_context, target, *_) in enumerate(self.test_loader):
                context, src, next_context, target, masks = self.__data_to_model(context, src, next_context, target)
                n_token = (target[1] != self.pad_idx).sum()
                # new model
                # if 'p' in self.args.data:
                #    output = self.model(context, src,  next_context, target[0], masks, 1)
                # else:
                #    output = self.model(context, src,  next_context, target[0], masks, 0)

                # if 'spn' in self.args.data or 'a' in self.args.data:
                output = self.model(context, src, next_context, target[0], masks, 3)
                # elif 'p' in self.args.data:
                #     output = self.model(context, src, next_context, target[0], masks, 1)
                # elif 'n' in self.args.data:
                #     output = self.model(context, src, next_context, target[0], masks, 2)
                # elif 's' in self.args.data:
                #     output = self.model(context, src, next_context, target[0], masks, 0)

                loss = self.scheduler(output, target[1], n_token)
                total_loss += (loss / n_token).item()
            total_loss_norm = total_loss / len(self.test_loader)
            print(f'gpu({self.args.gpu}),epoch({epoch}): val loss: {total_loss_norm}') if self.args.verbose else False

        # # only minimal loss and report 1 round for 3 epoches False: #
        # if False: #minloss < total_loss_norm and epoch % 3 != 0:
        #     eval_scores = {
        #         'eval_loss': total_loss_norm,
        #         'bleu4': 0,
        #         'meteor': 0,
        #         'rouge_l': 0,
        #         'cider': 0
        #     }
        #     return eval_scores

        # if min_loss > total_loss_norm:
        #     min_loss = total_loss_norm

        self.__predict(epoch)
        result_file = self.__merge_file(epoch)
        if result_file:
            metric = self.__calculate_scores(result_file, True)
            bleu4 = metric['Average across tIoUs']['Bleu_4'] * 100
            meteor = metric['Average across tIoUs']['METEOR'] * 100
            rouge_l = metric['Average across tIoUs']['ROUGE_L'] * 100
            cider = metric['Average across tIoUs']['CIDEr'] * 100

            eval_scores = {
                'eval_loss': total_loss_norm,
                'bleu4': bleu4,
                'meteor': meteor,
                'rouge_l': rouge_l,
                'cider': cider
            }
            return eval_scores
        return None

    def __predict(self, epoch):
        vocab = self.train_loader.dataset.caption_vocab.itos
        self.model.eval()
        predictions = {}

        for i, (context, src, next_context, target, _,_,(video_id, start, end)) in enumerate(self.test_loader):
            context, src, next_context, target, masks = self.__data_to_model(context, src, next_context, target)
            predict_tensor = self.__decoder(context, src, next_context, masks).long().cpu()
            predict_words = [[vocab[each_word] for each_word in each_instance] for each_instance in predict_tensor]
            for idx, each_video in enumerate(video_id):
                curr_start = start[idx]
                curr_end = end[idx]
                sent = predict_words[idx]
                if vocab[self.eos_idx] in sent:
                    sent = sent[:sent.index(vocab[self.eos_idx])]
                sent = ' '.join(sent[1:]).capitalize()
                if each_video not in predictions:
                    predictions[each_video] = []
                predictions[each_video].append(
                    {"sentence": sent, "timestamp": [curr_start.item(), curr_end.item()]}
                )
        result_partition_path = os.path.join(self.args.log_path, 'partition', f'e{epoch:02d}_{self.args.gpu:02d}.pt')
        pickle.dump(predictions, open(result_partition_path, 'wb'))

    def __merge_file(self, epoch):
        file_list = os.listdir(os.path.join(self.args.log_path, 'partition'))
        partitions = [each for each in file_list if re.match(f'e{epoch:02d}', each)]
        if len(partitions) != self.args.world_size:
            return None
        predictions = {}
        for each in partitions:
            curr_file = os.path.join(os.path.join(self.args.log_path, 'partition', each))
            curr = pickle.load(open(curr_file, 'rb'))
            for each_video in curr:
                if each_video not in predictions:
                    predictions[each_video] = []
                predictions[each_video] += curr[each_video]
        result_file = os.path.join(self.args.log_path, 'inferences', f'result_e{epoch:02d}.json')
        json.dump({'results': predictions}, open(result_file, 'w'))
        return result_file

    def __calculate_scores(self, result_file, block_print=True):
        if block_print:
            text_trap = io.StringIO()
            sys.stdout = text_trap
        metrics = {}
        prediction_fields = ['results']
        evaluator = ANETcaptions(
            self.args.reference_paths, result_file, self.args.tIoUs,
            1000, prediction_fields, False)
        evaluator.evaluate()

        for i, tiou in enumerate(self.args.tIoUs):
            metrics[tiou] = {}

            for metric in evaluator.scores:
                score = evaluator.scores[metric][i]
                metrics[tiou][metric] = score

        metrics['Average across tIoUs'] = {}
        for metric in evaluator.scores:
            score = evaluator.scores[metric]
            metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))
        sys.stdout = sys.__stdout__
        return metrics

    def __decoder(self, context, src,  next_context, masks):
        curr_batch_size = src[0].size(0)
        complete = torch.zeros(curr_batch_size, 1).byte().to(self.args.gpu)
        target = (torch.ones(curr_batch_size, 1) * self.init_idx).long().cuda(self.args.gpu, non_blocking=True)
        with torch.no_grad():
            while target.size(1) < self.args.seq_len and not all(complete):
                target_mask = utils.target_mask(target, self.pad_idx)
                masks = masks[:3] + (target_mask,)
                # if 'p' in self.args.data:
                #     output = self.model(context, src,  next_context, target, masks, 1)
                # else:
                #     output = self.model(context, src,  next_context, target, masks, 0)
                if 'spn' in self.args.data or 'a' in self.args.data:
                    output = self.model(context, src, next_context, target, masks, 3)
                elif 'p' in self.args.data:
                    output = self.model(context, src, next_context, target, masks, 1)
                elif 'n' in self.args.data:
                    output = self.model(context, src, next_context, target, masks, 2)
                elif 's' in self.args.data:
                    output = self.model(context, src, next_context, target, masks, 0)

                next_word = output[:, -1].max(dim=-1)[1].unsqueeze(1)
                target = torch.cat([target, next_word], dim=-1)
                complete = complete | torch.eq(next_word, self.eos_idx).byte()
        return target

    def __allocate_tensor_to_gpu(self, context, src, next_context, target):

        if self.args.gpu is not None:
            context = [each.cuda(self.args.gpu, non_blocking=True) for each in context]
            next_context = [each.cuda(self.args.gpu, non_blocking=True) for each in next_context]
            src = [each.cuda(self.args.gpu, non_blocking=True) for each in src]
        if torch.cuda.is_available():
            target = target.cuda(self.args.gpu, non_blocking=True)

        b = target.size(0)
        for i in range(target.size(1)):
            if all(target[:, i] == self.pad_idx):
                target = target[:, :i]
                break
        model_target = target[:, :-1]
        loss_target = target[:, 1:]
        return context, src, next_context, (model_target, loss_target)

    def __data_to_model(self, context, src, next_context, target):
        context, src, next_context, target = self.__allocate_tensor_to_gpu(context, src, next_context, target)

        if not self.args.context:
            context_text_mask = None
            context_audio_mask = None
            context_video_mask = None
            next_context_text_mask = None
            next_context_audio_mask = None
            next_context_video_mask = None

        elif self.args.context == 'seg_caption':
            context_text_mask = utils.data_mask(context[0], self.pad_idx)
            context_audio_mask = context_text_mask.data.clone()
            context_video_mask = context_text_mask.data.clone()
            next_context_text_mask = utils.data_mask(next_context[0], self.pad_idx)
            next_context_audio_mask = next_context_text_mask.data.clone()
            next_context_video_mask = next_context_text_mask.data.clone()

        elif self.args.context in ['frame', 'seg_frame']:
            context_text_mask = utils.data_mask(context[0], self.pad_idx)
            context_audio_mask = utils.data_mask(context[1][:, :, 0], self.pad_idx)
            context_video_mask = utils.data_mask(context[2][:, :, 0], self.pad_idx)
            next_context_text_mask = utils.data_mask(next_context[0], self.pad_idx)
            next_context_audio_mask = utils.data_mask(next_context[1][:, :, 0], self.pad_idx)
            next_context_video_mask = utils.data_mask(next_context[2][:, :, 0], self.pad_idx)

        else:
            raise ValueError(f'Unknown context : {self.args.context}')

        src_text_mask = utils.data_mask(src[0], self.pad_idx)
        src_audio_mask = utils.data_mask(src[1][:, :, 0], self.pad_idx)
        src_video_mask = utils.data_mask(src[2][:, :, 0], self.pad_idx)
        target_mask = utils.target_mask(target[0], self.pad_idx)

        context_masks = (context_text_mask, context_audio_mask, context_video_mask)
        src_masks = (src_text_mask, src_audio_mask, src_video_mask)
        next_context_masks = (next_context_text_mask, next_context_audio_mask, next_context_video_mask)
        masks = context_masks, src_masks, next_context_masks, target_mask

        return context, src, next_context, target, masks




