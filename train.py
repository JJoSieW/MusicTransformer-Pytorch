import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss, ContourLoss, CombinedLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import train_epoch, eval_model

from tqdm import tqdm
import json
import matplotlib.pyplot as plt


CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Val loss", "Val accuracy"]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_args()
    print_train_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter

        tensorboad_dir = os.path.join(args.output_dir, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    ##### Continuing from previous training session #####
    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    ##### Lr Scheduler vs static lr #####
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr


    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    # if(args.ce_smoothing is None):
    #     train_loss_func = eval_loss_func
    # else:
    #     train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)
    
    # ignore_indices: for SmoothCrossEntropyLoss
    ignore_indices = list(range(TOKEN_NOTE, TOKEN_NOTE + TOKEN_CONTOUR)) + [TOKEN_PAD]
    
    smooth_entropy_loss = SmoothCrossEntropyLoss(
        label_smoothing=args.ce_smoothing,
        vocab_size=VOCAB_SIZE,
        ignore_index=ignore_indices
    )

    contour_aware_loss = ContourLoss(margin=0.87)
    
    # 原设定：前50个epoch才增长完成
    # 更实用的 lambda_schedule（收敛前稳定）：
    # 优点：训练中期（第15轮）λ 就稳定，contour loss 提前发挥；
	# •	兼顾收敛和泛化：也不会训练一开始就干扰 token-level 预测；
	# •	适合你当前 30~50 epoch 的训练周期。
    def lambda_scheduler(epoch):
        warmup_epochs = 15
        max_lambda = 0.3
        if epoch <= warmup_epochs:
            return (epoch / warmup_epochs) * max_lambda
        else:
            return max_lambda
    
    train_loss_func = CombinedLoss(
        ce_loss_fn=smooth_entropy_loss,
        contour_loss_fn=contour_aware_loss,
        lambda_contour_scheduler=lambda_scheduler
    )

    eval_loss_func = train_loss_func



    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    # 添加总体训练进度条
    epoch_range = range(start_epoch, args.epochs)
    epoch_pbar = tqdm(epoch_range, desc='Training Progress', unit='epoch', leave=True)
    
    train_loss_curve = []
    val_loss_curve = []
    train_acc_curve = []
    val_acc_curve = []
    
    for epoch in epoch_pbar:
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if(epoch > BASELINE_EPOCH):
            epoch_pbar.set_postfix({'Phase': 'Training'})
            
            # Train
            train_epoch(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, args.print_modulus)

            epoch_pbar.set_postfix({'Phase': 'Evaluating'})
        else:
            epoch_pbar.set_postfix({'Phase': 'Baseline Eval'})

        # Validation
        train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        val_loss, val_acc = eval_model(model, val_loader, eval_loss_func)

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)

        # Learn rate
        lr = get_lr(opt)

        # 更新总体进度条信息
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_acc:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'LR': f'{lr:.6f}'
        })

        print("Epoch:", epoch+1)
        print("Avg train loss:", train_loss)
        print("Avg train acc:", train_acc)
        print("Avg val loss:", val_loss)
        print("Avg val acc:", val_acc)
        print(SEPERATOR)
        print("")

        new_best = False

        if(val_acc > best_eval_acc):
            best_eval_acc = val_acc
            best_eval_acc_epoch  = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if(val_loss < best_eval_loss):
            best_eval_loss       = val_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
                print("Best eval acc:", best_eval_acc, file=o_stream)
                print("")
                print("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best eval loss:", best_eval_loss, file=o_stream)


        if(not args.no_tensorboard):
            tensorboard_summary.add_scalar("Avg_combined_loss/train", train_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_combined_loss/val", val_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/train", train_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/val", val_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Learn_rate/train", lr, global_step=epoch+1)
            tensorboard_summary.flush()

        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_acc, val_loss, val_acc])

    # Save loss curves to JSON
    loss_curve_file = os.path.join(args.output_dir, "loss_curve.json")
    with open(loss_curve_file, "w") as f:
        json.dump({
            "train_loss": train_loss_curve,
            "val_loss": val_loss_curve,
            "train_acc": train_acc_curve,
            "val_acc": val_acc_curve
        }, f)


    # Plot loss and accuracy curves using matplotlib

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot loss curves
    ax1.plot(train_loss_curve, label='Train Loss')
    ax1.plot(val_loss_curve, label='Validation Loss') 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy curves
    ax2.plot(train_acc_curve, label='Train Accuracy')
    ax2.plot(val_acc_curve, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'learning_curves.png'))
    plt.close()


    # Sanity check just to make sure everything is gone
    if(not args.no_tensorboard):
        tensorboard_summary.flush()

    # 在代码中查看
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    return


if __name__ == "__main__":
    main()
