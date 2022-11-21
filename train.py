import torch
import time
import os
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image
from kvasir_dataset import (
    get_transforms,
    get_dataset,
    get_dataloader,
    check_dataloader
)


def get_model():
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model


def log_val(model, best_metric, writer, epoch, metric):
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), "best.pth")
        print("saved new best metric model")

    print(
        "current epoch: {}\
                    current mean dice: {:.4f}\
                    best mean dice: {:.4f}\
                    at epoch {}".
        format(epoch + 1, metric, best_metric, best_metric_epoch)
    )

    writer.add_scalar("val_mean_dice", metric, epoch + 1)
    return best_metric, best_metric_epoch


def log_train(train_loader, writer, epoch, epoch_loss, step, loss):
    epoch_loss += loss.item()
    epoch_len = len(train_loader) // train_loader.batch_size
    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    writer.add_scalar("train_loss", loss.item(),
                      epoch_len * epoch + step)


def train(model, device, loss_function, optimizer, lr_scheduler, batch_data):
    inputs, labels = batch_data["img"].to(
        device), batch_data["seg"].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    lr_scheduler.step()
    return loss


def val(val_loader, model, device, post_trans, dice_metric, metric_values):
    val_images = None
    val_labels = None
    val_outputs = None
    for val_data in val_loader:
        val_images, val_labels = val_data["img"].to(
            device), val_data["seg"].to(device)
        roi_size = (96, 96)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_images, roi_size, sw_batch_size, model)
        val_outputs = [post_trans(i)
                       for i in decollate_batch(val_outputs)]
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()
    metric_values.append(metric)
    return val_images, val_labels, val_outputs, metric


def setup_hyperparams(model):
    max_epochs = 2
    val_interval = 2
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5,
                             squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    return max_epochs,val_interval,loss_function,optimizer,lr_scheduler,dice_metric


# VAL_AMP = True
# def inference(input, model):
#     def _compute(input):
#         return sliding_window_inference(
#             inputs=input,
#             roi_size=(240, 240, 160),
#             sw_batch_size=1,
#             predictor=model,
#             overlap=0.5,
#         )

#     if VAL_AMP:
#         with torch.cuda.amp.autocast():
#             return _compute(input)
#     else:
#         return _compute(input)
        
def main():
    
    KSEG_PATH = "/home/htluc/datasets/Kvasir-SEG"
    root_dir = "/home/htluc/vocalfold/"
    device = torch.device("cuda:0")
    
    train_transforms, val_transforms = get_transforms()

    train_ds = get_dataset(KSEG_PATH, train_transforms)
    val_ds = get_dataset(KSEG_PATH, val_transforms)

    train_loader, val_loader = get_dataloader(train_ds, val_ds)

    check_dataloader(train_loader)
    check_dataloader(val_loader)

    model = get_model()
    model = model.to(device)
    
    # Define loss function and optimizer, lr_scheduler, and metrics.
    post_trans = Compose([
        Activations(sigmoid=True), 
        AsDiscrete(threshold=0.5)
    ])
    
    max_epochs, val_interval, loss_function, optimizer, lr_scheduler, dice_metric = setup_hyperparams(model)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1:2d}/{max_epochs:2d}")

        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            # Process a mini-batch
            step += 1
            loss = train(model, device, loss_function, optimizer,
                         lr_scheduler, step, batch_data)

            # Log down
            log_train(train_loader, writer, epoch, epoch_loss, step, loss)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1:2d} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images, val_labels, val_outputs, metric = val(
                    val_loader, model, device, post_trans, dice_metric, metric_values)

                best_metric, best_metric_epoch = log_val(
                    model, best_metric, writer, epoch, metric)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # plot_2d_or_3d_image(val_images, epoch + 1,
                #                     writer, index=0, tag="image")
                # plot_2d_or_3d_image(val_labels, epoch + 1,
                #                     writer, index=0, tag="label")
                # plot_2d_or_3d_image(val_outputs, epoch + 1,
                #                     writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()