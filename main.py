import sys
import os
import warnings
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

import config
from models import SimpleEncoderDecoderCat, UnifiedTriEncoderTransformer
from dataloader import MyDataset
from loss import Scheduler
from utils import save_checkpoint
from runner import EpochRunner


best_meteor = 0.0
scores_record = {
    'epoch': [],
    'train_loss': [],
    'eval_loss': [],
    'bleu4': [],
    'meteor': [],
    'rouge_l': [],
    'cider': []
}


def main():
    args = config.parse_config()

    timestamp = int(time.time())
    args.log_path = os.path.join(args.log_path, f'{timestamp}')

    # make log dir
    os.makedirs(args.log_path)
    os.makedirs(os.path.join(args.log_path, 'model'))
    os.makedirs(os.path.join(args.log_path, 'inferences'))
    os.makedirs(os.path.join(args.log_path, 'partition'))
    pickle.dump(args, open(os.path.join(args.log_path, 'config.pt'), 'wb'))

    fo = open(os.path.join(args.log_path, 'parameters.txt'), "w")
    for i in range(0, len(sys.argv)):
        fo.write(sys.argv[i])
    fo.close()

    print(f'log path : {args.log_path}') if args.verbose else False

    # set seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.warm_up == -1:
        args.warm_up = int(args.epoch/2)

    n_gpus_per_node = torch.cuda.device_count()
    args.world_size = n_gpus_per_node * args.world_size

    # construct communication tunnel
    if 'file:///' in args.dist_url:
        timestamp = int(time.time())
        args.dist_url += f'_{timestamp}'

    # start distributed running
    if args.world_size > 1 and args.gpu is None:
        mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, args))
    else:
        args.world_size = 1
        main_worker(args.gpu, 1, args)


def main_worker(gpu, n_gpus_per_node, args):

    global best_meteor
    global scores_record

    # distribute init
    print(f"Use GPU: {gpu} for training") if args.verbose and gpu is not None else False

    distributed = args.world_size > 1
    args.gpu = gpu
    if distributed:
        args.rank = args.rank * n_gpus_per_node + gpu
        dist.init_process_group('nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # dataset
    print('loading training dataset') if args.verbose else False
    train_dataset = MyDataset(
        args.train_dataset, args.text_feature, args.audio_feature, args.video_feature,
        None, args.min_freq, args.modality,
        max_len=args.seq_len, context_len= args.context_len,context=args.context, on_memory=args.on_memory
    )
    print('loading validation dataset') if args.verbose else False
    test_dataset = MyDataset(
        args.test_dataset, args.text_feature, args.audio_feature, args.video_feature,
        train_dataset, args.min_freq, args.modality,
        max_len=args.seq_len, context_len= args.context_len, context=args.context, on_memory=args.on_memory
    )

    # model
    print('loading model') if args.verbose else False
    if args.model == 'base':
        target_model = SimpleEncoderDecoderCat
    elif args.model == 'uni_tricoder':
        target_model = UnifiedTriEncoderTransformer
    else:
        raise ValueError(f'Unknown model : {args.model}')

    dim_feature = args.dim_audio if args.modality == 'a' else args.dim_video
    model = target_model(
        len(train_dataset.caption_vocab), dim_feature, args.dim_model, args.dim_ff,
        args.head, args.n_layer, args.dropout, args.modality,
        n_src_vocab=len(train_dataset.text_vocab),
        args=args
    )

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters : {total_params}") if args.verbose else False
    print(f"trainable parameters : {trainable_params}") if args.verbose else False

    if distributed:
        args.batch_size = args.batch_size // n_gpus_per_node
        args.n_worker = (args.n_worker + n_gpus_per_node - 1) // n_gpus_per_node
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        torch.backends.cudnn.benchmark = True

    # dataloader
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.n_worker, sampler=train_sampler, shuffle=train_sampler is None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.n_worker,
    )

    # scheduler
    print('loading scheduler') if args.verbose else False
    scheduler = Scheduler(model, train_dataset.pad_idx, args)

    # epoch runner
    print('loading epoch runner') if args.verbose else False
    trainer = EpochRunner(model, train_loader, test_loader, scheduler, args)

    min_loss = float('inf')

    # run epoch
    for i in range(args.epoch):
        if train_sampler:
            train_sampler.set_epoch(i)
        loss = trainer.train(i)

        scores_record['epoch'].append(i)
        scores_record['train_loss'].append(loss)
    
        if i < args.warm_up:
            scores_record['eval_loss'].append(0)
            scores_record['bleu4'].append(0)
            scores_record['meteor'].append(0)
            scores_record['rouge_l'].append(0)
            scores_record['cider'].append(0)
            continue

        scores = trainer.eval(i, min_loss)
        min_loss = max(min_loss, scores['eval_loss'])

        if scores:
            best_meteor = max(best_meteor, scores['meteor'])
            is_best = best_meteor == scores['meteor']
            if args.save_model and (i % args.log_freq == 0 or is_best):
                save_checkpoint({
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'scores': scores,
                    'optimizer': scheduler.optimizer.state_dict(),
                }, is_best, i, args.log_path)
            print('**************************************************************')
            print(f'epoch({i}): scores {scores}') if args.verbose else False
            print('**************************************************************')

            for each in scores:
                scores_record[each].append(scores[each])
            if scores['bleu4'] != 0:
                record_path = os.path.join(args.log_path, 'score_record'+str(i)+'.csv')
                pd.DataFrame(scores_record).to_csv(record_path)
    print(f'best_meteor : {best_meteor}')


if __name__ == '__main__':
    main()
