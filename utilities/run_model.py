import torch
import time
from tqdm import tqdm

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy

# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """
    if hasattr(loss, "set_epoch"):
        loss.set_epoch(cur_epoch)

    out = -1
    
    model.train()
    
    # 添加进度条
    pbar = tqdm(dataloader, desc=f'Epoch {cur_epoch} Training', 
                unit='batch', leave=True)
    
    for batch_num, batch in enumerate(pbar):
        
        time_before = time.time()

        opt.zero_grad()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())

        y = model(x)  
        # y - logits: [B, T, V]
        # tgt - targets: [B, T]    ------ B: batch size, T: sequence length, V: vocabulary size
        
        out = loss.forward(y, tgt)
        out.backward()
        opt.step()

        if(lr_scheduler is not None):
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        # 更新进度条信息
        pbar.set_postfix({
            'LR': f'{get_lr(opt):.6f}',
            'Loss': f'{float(out):.4f}',
            'Time': f'{time_took:.2f}s'
        })

        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")

    return

# eval_model
def eval_model(model, dataloader, loss):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    
    # 添加进度条
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch', leave=True)
    
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        
        for batch_num, batch in enumerate(pbar):
            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y = model(x)

            batch_acc = float(compute_epiano_accuracy(y, tgt))
            sum_acc += batch_acc

            # y   = y.reshape(y.shape[0] * y.shape[1], -1)
            # tgt = tgt.flatten()

            out = loss.forward(y, tgt)
            batch_loss = float(out)
            sum_loss += float(out)

            # 更新进度条信息
            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_acc:.4f}'
            })

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc
