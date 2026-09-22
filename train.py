import os
import random
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler

from config import *
from utils.dataset import build_dataset
from utils.loss import loss_fn
from utils.metrics import get_results
from utils.qmodel import QualityModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_quality_model():
    q_model = QualityModel(model_name=QMODEL, num_classes=Q_LEVELS, device=DEVICE)
    q_model.load_state_dict(torch.load(QUALITY_MODEL_PATH, map_location=DEVICE))
    q_model.eval()
    return q_model


def build_segmentation_model():
    model = smp.Unet(BACKBONE, encoder_weights=PRE_TRAINED, classes=1, activation='sigmoid')
    return model.to(DEVICE)


def run_epoch(model, q_model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    for step, batch in enumerate(loader):
        images = batch['image'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        with torch.set_grad_enabled(training):
            with torch.no_grad():
                quality_probs = q_model(images)
            preds = model(images)
            loss = loss_fn(preds, labels, quality_probs, loss_type=LOSS_TYPE, weight_mode=WEIGHT_MODE)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        print(f'[{step + 1}/{len(loader)}] loss: {loss.item():.4f}', end='\r')

    iou, dice, precision, recall = get_results(np.concatenate(all_preds), np.concatenate(all_labels))
    return {'loss': total_loss / len(loader), 'iou': iou, 'dice': dice,
            'precision': precision, 'recall': recall}


def main():
    set_seed(SEED)

    run_name = f"{time.strftime('%Y%m%d-%H%M')}-{BACKBONE}-{STRUCTURE}-{LOSS_TYPE}-{WEIGHT_MODE}"
    save_dir = os.path.join(SAVE_PATH, run_name)
    weight_dir = os.path.join(save_dir, 'weights')
    os.makedirs(weight_dir, exist_ok=True)

    dataset_builder = build_dataset(image_size=IMGSIZE, batch_size=BATCH)
    train_loader = dataset_builder.train(TRAIN_PATH, augmentation=AUGMENTATION,
                                         transform_rate=TRANSFORM_RATE, num_workers=NUM_WORKERS)
    valid_loader = dataset_builder.valid(VALID_PATH)
    test_loader = dataset_builder.test(TEST_PATH)

    q_model = build_quality_model()
    model = build_segmentation_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_dice = 0.0
    best_weight_path = os.path.join(weight_dir, 'best.pth')
    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} | lr: {optimizer.param_groups[0]['lr']:.2e}")

        train_result = run_epoch(model, q_model, train_loader, optimizer)
        valid_result = run_epoch(model, q_model, valid_loader)
        scheduler.step()

        print(f"train loss: {train_result['loss']:.4f} dice: {train_result['dice']:.4f} iou: {train_result['iou']:.4f}")
        print(f"valid loss: {valid_result['loss']:.4f} dice: {valid_result['dice']:.4f} iou: {valid_result['iou']:.4f}")

        history.append({'epoch': epoch + 1,
                        **{f'train_{k}': v for k, v in train_result.items()},
                        **{f'valid_{k}': v for k, v in valid_result.items()}})
        pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_info.csv'), index=False)

        if valid_result['dice'] > best_dice:
            best_dice = valid_result['dice']
            torch.save(model.state_dict(), best_weight_path)
            print(f'saved best model (valid dice: {best_dice:.4f})')

    model.load_state_dict(torch.load(best_weight_path, map_location=DEVICE))
    test_result = run_epoch(model, q_model, test_loader)
    print(f"\ntest dice: {test_result['dice']:.4f} iou: {test_result['iou']:.4f} "
          f"precision: {test_result['precision']:.4f} recall: {test_result['recall']:.4f}")
    pd.DataFrame([test_result]).to_csv(os.path.join(save_dir, 'test_result.csv'), index=False)


if __name__ == '__main__':
    main()
