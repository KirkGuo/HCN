import torch
import shutil
import os


def data_mask(x, pad_idx):
    mask = (x != pad_idx).unsqueeze(1)
    return mask


def target_mask(x, pad_idx):
    mask = (x != pad_idx).unsqueeze(-2)
    b, l = x.size()
    mask = mask & torch.tril(torch.ones(1, l, l), 0).byte().type_as(mask.data)
    return mask


def save_checkpoint(state, is_best, epoch, save_path):
    filename = f'checkpoint_e{epoch:02d}.pth.tar'
    save_name = os.path.join(save_path, 'model', filename)
    torch.save(state, save_name)
    if is_best:
        shutil.move(save_name, os.path.join(save_path, 'model_best.pth.tar'))


def __test():
    dummy_data = [1, 2, 14, 6, 3, 0, 0, 0, 0, 0]
    dummy_target = [1, 345, 33, 21, 4, 5, 5, 3, 0, 0]
    dummy_data = torch.tensor(dummy_data).unsqueeze(0)
    dummy_target = torch.tensor(dummy_target).unsqueeze(0).long()
    dummy_data_mask = data_mask(dummy_data, 0)
    dummy_target_mask = target_mask(dummy_target, 0)
    print(dummy_data_mask)
    print(dummy_target_mask)


if __name__ == '__main__':
    __test()
