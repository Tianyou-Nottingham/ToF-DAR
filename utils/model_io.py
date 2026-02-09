import os
import torch


def get_state_dict_model(model):
    return model.module if hasattr(model, "module") else model


def save_weights(model, fpath):
    folder = os.path.dirname(fpath)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    torch.save(get_state_dict_model(model).state_dict(), fpath)
    return


def load_weights(model, fpath):
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, optimizer, epoch, fpath, global_step=None):
    folder = os.path.dirname(fpath)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if global_step is not None:
        payload["global_step"] = int(global_step)

    torch.save(payload, fpath)


def load_checkpoint(model, fpath, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model, optimizer, epoch
