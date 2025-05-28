import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score # Added accuracy_score
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Use standard tqdm
import random
import time
import sys

PROCESSED_DATASET_BASE_PATH = '/home/vaibhav/PROJECTS/processed_dataset_faces_mediapipe'

MODEL_SAVE_PATH = '/home/vaibhav/PROJECTS/deepfake_models_output_local'

SEQUENCE_LENGTH = 20
IMG_SIZE = 112
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class VideoDataset(Dataset):
    def __init__(self, video_files_list, labels_list, sequence_length, transform, img_size):
        self.video_files = video_files_list
        self.labels = labels_list
        self.sequence_length = sequence_length
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices_to_sample = []
        if total_frames_in_video >= self.sequence_length:
            indices_to_sample = np.linspace(0, total_frames_in_video - 1, self.sequence_length, dtype=int)
        else:
            indices_to_sample = np.arange(total_frames_in_video)
        frames_read = 0
        current_frame_idx = 0
        while frames_read < len(indices_to_sample) and len(frames) < self.sequence_length :
            ret, frame = cap.read()
            if not ret: break
            if current_frame_idx == indices_to_sample[frames_read]:
                frames.append(self.transform(frame))
                frames_read += 1
            current_frame_idx += 1
        cap.release()
        while len(frames) < self.sequence_length:
            if not frames:
                dummy_frame_np = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(self.transform(dummy_frame_np))
            else: frames.append(frames[-1])
        frames_tensor = torch.stack(frames)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=512, bidirectional_lstm=False, dropout_rate=0.5):
        super(DeepFakeDetector, self).__init__()
        cnn_model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.cnn_features = nn.Sequential(*list(cnn_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=lstm_layers, bidirectional=bidirectional_lstm, batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional_lstm else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x_cnn = x.view(batch_size * seq_length, c, h, w)
        cnn_out = self.cnn_features(x_cnn)
        pooled_out = self.avgpool(cnn_out)
        flattened_out = pooled_out.view(batch_size * seq_length, -1)
        lstm_input = flattened_out.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(lstm_input)
        final_lstm_out = torch.mean(lstm_out, dim=1)
        out = self.dropout(final_lstm_out)
        out = self.fc(out)
        return out

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

def train_epoch(model, data_loader, criterion, optimizer, device, epoch_num, total_epochs):
    model.train()
    losses = AverageMeter(); accuracies = AverageMeter()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Train]", leave=False, file=sys.stdout)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = calculate_accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        progress_bar.set_postfix(loss=losses.avg, acc=f"{accuracies.avg*100:.2f}%")
    return losses.avg, accuracies.avg

def validate_epoch(model, data_loader, criterion, device, epoch_num, total_epochs):
    model.eval()
    losses = AverageMeter(); accuracies = AverageMeter()
    all_labels_val = []; all_preds_probs_val = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Valid]", leave=False, file=sys.stdout)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            all_labels_val.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds_probs_val.extend(probs)
            progress_bar.set_postfix(loss=losses.avg, acc=f"{accuracies.avg*100:.2f}%")
    auc = roc_auc_score(all_labels_val, all_preds_probs_val) if len(set(all_labels_val)) > 1 else 0.0
    return losses.avg, accuracies.avg, auc

def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_real_files = sorted(glob.glob(os.path.join(PROCESSED_DATASET_BASE_PATH, 'train/real_faces/*.mp4')))
    train_fake_files = sorted(glob.glob(os.path.join(PROCESSED_DATASET_BASE_PATH, 'train/fake_faces/*.mp4')))
    if not train_real_files and not train_fake_files:
        print(f"ERROR: No processed training videos found in {PROCESSED_DATASET_BASE_PATH}/train/. Please run preprocessing first.")
        return
    all_train_video_files = train_real_files + train_fake_files
    all_train_labels = [1] * len(train_real_files) + [0] * len(train_fake_files)
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_train_video_files, all_train_labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=all_train_labels)

    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    train_dataset = VideoDataset(train_files, train_labels, SEQUENCE_LENGTH, train_transforms, IMG_SIZE)
    val_dataset = VideoDataset(val_files, val_labels, SEQUENCE_LENGTH, val_transforms, IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=max(0,os.cpu_count()//2 -1), pin_memory=True if device.type == 'cuda' else False) # Adjust num_workers
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=max(0,os.cpu_count()//2 -1), pin_memory=True if device.type == 'cuda' else False)

    model = DeepFakeDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_loss_history, train_acc_history, val_loss_history, val_acc_history, val_auc_history = [], [], [], [], []
    best_val_auc = 0.0
    model_filename_base = f'deepfake_detector_local_seq{SEQUENCE_LENGTH}'
    best_model_path = os.path.join(MODEL_SAVE_PATH, f'{model_filename_base}_best_auc.pth')

    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, device, epoch, NUM_EPOCHS)
        train_loss_history.append(train_loss); train_acc_history.append(train_acc)
        val_loss_history.append(val_loss); val_acc_history.append(val_acc); val_auc_history.append(val_auc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"   Best model saved to {best_model_path} (Val AUC: {best_val_auc:.4f})")
    end_time = time.time()
    print(f"--- Training Finished in {(end_time - start_time)/60:.2f} minutes ---")
    print(f"Best Validation AUC: {best_val_auc:.4f}, Model at: {best_model_path}")
    last_model_path = os.path.join(MODEL_SAVE_PATH, f'{model_filename_base}_last_epoch.pth')
    torch.save(model.state_dict(), last_model_path)
    print(f"Last epoch model saved to {last_model_path}")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(train_loss_history, label='Train Loss'); plt.plot(val_loss_history, label='Val Loss'); plt.legend(); plt.title('Loss'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot([a*100 for a in train_acc_history], label='Train Acc'); plt.plot([a*100 for a in val_acc_history], label='Val Acc'); plt.legend(); plt.title('Accuracy (%)'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(val_auc_history, label='Val AUC'); plt.legend(); plt.title('AUC'); plt.grid(True)
    plt.tight_layout()
    plot_save_path = os.path.join(MODEL_SAVE_PATH, f'{model_filename_base}_training_plots.png')
    plt.savefig(plot_save_path); print(f"Training plots: {plot_save_path}"); plt.show()

if __name__ == '__main__':
    main()