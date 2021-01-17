import argparse


def parse_config():
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--text_feature", type=str)
    parser.add_argument("--audio_feature", type=str)
    parser.add_argument("--video_feature", type=str)

    # experiment config
    parser.add_argument("--context", choices=['', 'frame', 'seg_frame', 'seg_caption'])
    parser.add_argument("--model", choices=['base', 'uni-base','gate', 'uni-gate', 'uni_bicoder', 'uni_tricoder'])
    parser.add_argument("--modality", choices=['t', 'a', 'v', 'vt','va'])
    parser.add_argument("--data", choices=['s', 'p', 'pn', 'sp', 'sn','sa', 'spn'])

    # model config
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dim_audio", type=int)
    parser.add_argument("--dim_video", type=int)
    parser.add_argument("--dim_model", type=int)
    parser.add_argument("--dim_ff", type=int)
    parser.add_argument("--head", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--min_freq", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--context_len", type=int)

    # dataloader config
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--n_worker", type=int)
    parser.add_argument("--on_memory", type=bool)

    # optimizer config
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_decay", type=float)
    parser.add_argument("--adam_weight_decay", type=float)
    parser.add_argument("--adam_beta1", type=float)
    parser.add_argument("--adam_beta2", type=float)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--warm_up", type=int)

    # evaluation config
    parser.add_argument("--tIoUs", type=float, nargs='+')
    parser.add_argument("--reference_paths", type=str, nargs='+')

    # hardware config
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cuda_devices", type=int, nargs='+')
    parser.add_argument("--dist_url", type=str)
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--rank", type=int)

    # log config
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--log", type=bool)
    parser.add_argument("--log_freq", type=int)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--save_model", type=bool)

    # set default args
    # parser.set_defaults(train_dataset='/s1_md0/leiji/activitynet/data/train.csv')
    # parser.set_defaults(test_dataset='/s1_md0/leiji/activitynet/data/val_sub.csv')
    # parser.set_defaults(text_feature='')
    # parser.set_defaults(audio_feature='/s1_md0/t-huluo/files/Activity_Coin_Cross/feature.audio.ghostvlad')
    # parser.set_defaults(video_feature='/s1_md0/t-huluo/files/Activity_Coin_Cross/feature.s3dg')
    parser.set_defaults(train_dataset='data/train.csv')
    parser.set_defaults(test_dataset='data/val.csv')
    parser.set_defaults(text_feature='/s1_md0/leiji/kirk_xlg/youcook/word_sequences.json')
    parser.set_defaults(audio_feature='/s1_md0/leiji/kirk_xlg/youcook/youcookii_audio_feature')
    parser.set_defaults(video_feature='/s1_md0/leiji/kirk_xlg/youcook/youcookii_videos_features3dg')
    parser.set_defaults(context='seg_frame')
    parser.set_defaults(model='uni_tricoder')
    parser.set_defaults(data='spn')
    parser.set_defaults(modality='vt')
    parser.set_defaults(seed=-1)
    parser.set_defaults(dim_audio=512)
    parser.set_defaults(dim_video=1024)
    parser.set_defaults(dim_model=512)
    parser.set_defaults(dim_ff=128)
    parser.set_defaults(head=8)
    parser.set_defaults(n_layer=1)
    parser.set_defaults(dropout=0.4)
    parser.set_defaults(min_freq=1)
    parser.set_defaults(seq_len=80)
    parser.set_defaults(context_len=10)
    parser.set_defaults(batch_size=64)
    parser.set_defaults(epoch=30)
    parser.set_defaults(n_worker=1)
    parser.set_defaults(on_memory=False)
    parser.set_defaults(lr=1e-4)
    parser.set_defaults(lr_decay=0.1)
    parser.set_defaults(adam_weight_decay=0.00)
    parser.set_defaults(adam_beta1=0.9)
    parser.set_defaults(adam_beta2=0.98)
    parser.set_defaults(smoothing=0.4)
    parser.set_defaults(warm_up=10)
    parser.set_defaults(tIoUs=[0.3, 0.5, 0.7, 0.9])
    # parser.set_defaults(reference_paths=['/s1_md0/leiji/activitynet/data/val_sub.json'])
    parser.set_defaults(reference_paths=['data/val.json'])
    parser.set_defaults(gpu=0)
    parser.set_defaults(cuda_devices=[0])  # now this parameter is not used
    parser.set_defaults(dist_url='tcp://127.0.0.1:9000')
    parser.set_defaults(world_size=1)
    parser.set_defaults(rank=0)
    parser.set_defaults(verbose=True)
    parser.set_defaults(log=False)
    parser.set_defaults(log_freq=5)
    parser.set_defaults(log_path='log')
    parser.set_defaults(save_model=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    test_args = parse_config()
    print(test_args.train_dataset)
    pass
