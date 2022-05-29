import numpy as np
import torch
import torch.nn.functional as F
from . import train_utils
from . import common_utils
from torch.cuda.amp import autocast

@common_utils.check_not_multihead(0)
def get_accuracy_and_loss(model, loader, device, loss_fn=F.cross_entropy, enable_amp=True):
    # loss_fn: any function that takes in (logits, targets) and outputs a scalar
    assert next(model.parameters()).device == device

    in_tr_mode = model.training
    model = model.eval()

    acc_meter = train_utils.AverageMeter()
    loss_meter = train_utils.AverageMeter()

    with torch.no_grad():
        for xb, yb, *_ in loader:
            bs = len(xb)
            xb, yb = xb.to(device), yb.to(device)

            with autocast(enabled=enable_amp):
                out = model(xb)

            preds = out.argmax(-1)

            b_acc = (preds==yb).float().mean().item()
            b_loss = loss_fn(out, yb).item()

            acc_meter.update(b_acc, bs)
            loss_meter.update(b_loss, bs)

            xb, yb = xb.cpu(), yb.cpu()

    if in_tr_mode:
        model.train()

    return {
        'acc': acc_meter.mean(),
        'loss': loss_meter.mean()
    }

@common_utils.check_not_multihead(0)
def get_confusion_matrix(model, loader, num_classes, device, enable_amp=True):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    model = model.eval()

    cmat = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device)

            with autocast(enabled=enable_amp):
                out = model(xb)

            yb = yb.cpu().clone()
            preds = out.argmax(-1).cpu().clone()

            for y, yh in zip(yb.numpy(), preds.numpy()):
                cmat[y][yh] += 1

            xb = xb.cpu()

    if in_tr_mode:
        model.train()

    return cmat

@common_utils.check_not_multihead(0)
def get_classwise_accuracies(model, loader, num_classes, device):
    cmat = get_confusion_matrix(model, loader, num_classes, device)
    return np.diag(cmat)/cmat.sum(axis=1)

@common_utils.check_not_multihead(0)
def get_predictions(model, loader, device, enable_amp=True):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    model = model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device)

            with autocast(enabled=enable_amp):
                out = model(xb)

            yh = out.argmax(-1).cpu().numpy()
            preds.append(yh)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

            xb = xb.cpu()

    if in_tr_mode:
        model.train()

    preds, labels = map(np.concatenate, [preds, labels])
    return preds, labels

@common_utils.check_not_multihead(0)
def get_margins(model, loader, device, enable_amp=True):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    model = model.eval()
    all_margins = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device, non_blocking=True)
            rng = torch.arange(len(xb))

            with autocast(enabled=enable_amp):
                out = model(xb)

            class_logits = out[rng, yb].clone()
            out[rng, yb] = -np.inf
            max_wo_class = out[rng, out.argmax(1)]
            class_logits = (class_logits - max_wo_class).cpu()
            all_margins.append(class_logits)

    if in_tr_mode:
        model = model.train()

    all_margins = torch.cat(all_margins).numpy()
    return all_margins