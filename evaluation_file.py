import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
import sys

PROCESSED_DATASET_BASE_PATH = '/home/vaibhav/PROJECTS/processed_dataset_faces_mediapipe'

TRAINED_MODEL_PATH = '/home/vaibhav/PROJECTS/deepfake_models_output_local/deepfake_detector_local_seq20_best_auc.pth'
SEQUENCE_LENGTH = 20
IMG_SIZE = 112
BATCH_SIZE = 8


class VideoDataset(Dataset):
    def __init__(self, video_files_list, labels_list, sequence_length, transform, img_size):
        self.video_files = video_files_list;
        self.labels = labels_list
        self.sequence_length = sequence_length;
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx];
        label = self.labels[idx]
        frames = [];
        cap = cv2.VideoCapture(video_path)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices_to_sample = []
        if total_frames_in_video >= self.sequence_length:
            indices_to_sample = np.linspace(0, total_frames_in_video - 1, self.sequence_length, dtype=int)
        else:
            indices_to_sample = np.arange(total_frames_in_video)
        frames_read = 0;
        current_frame_idx = 0
        while frames_read < len(indices_to_sample) and len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret: break
            if current_frame_idx == indices_to_sample[frames_read]:
                frames.append(self.transform(frame));
                frames_read += 1
            current_frame_idx += 1
        cap.release()
        while len(frames) < self.sequence_length:
            if not frames:
                dummy_frame_np = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(self.transform(dummy_frame_np))
            else:
                frames.append(frames[-1])
        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                 hidden_dim=512,  # Ensure this matches the trained model
                 bidirectional_lstm=False, # Ensure this matches the trained model
                 dropout_rate=0.5):
        super(DeepFakeDetector, self).__init__()
        cnn_model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        self.cnn_features = nn.Sequential(*list(cnn_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim,
                            num_layers=lstm_layers, bidirectional=bidirectional_lstm,
                            batch_first=True)
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

        lstm_out, _ = self.lstm(lstm_input) # Get all sequence outputs
        final_lstm_out = torch.mean(lstm_out, dim=1) # Take the mean across the sequence dimension

        out = self.dropout(final_lstm_out)
        out = self.fc(out)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_transforms = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_real_files = sorted(glob.glob(os.path.join(PROCESSED_DATASET_BASE_PATH, 'test/real_faces/*.mp4')))
    test_fake_files = sorted(glob.glob(os.path.join(PROCESSED_DATASET_BASE_PATH, 'test/fake_faces/*.mp4')))
    if not test_real_files and not test_fake_files:
        print(
            f"ERROR: No processed test videos found in {PROCESSED_DATASET_BASE_PATH}/test/. Please run preprocessing first.")
        return
    all_test_video_files = test_real_files + test_fake_files
    all_test_labels = [1] * len(test_real_files) + [0] * len(test_fake_files)
    print(f"Total test samples: {len(all_test_video_files)}")

    test_dataset = VideoDataset(all_test_video_files, all_test_labels, SEQUENCE_LENGTH, test_transforms, IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=max(0, os.cpu_count() // 2 - 1),
                             pin_memory=True if device.type == 'cuda' else False)

    model = DeepFakeDetector().to(device)
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"ERROR: Trained model not found at {TRAINED_MODEL_PATH}. Please check the path.")
        return
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {TRAINED_MODEL_PATH} and set to evaluation mode.")

    all_true_labels, all_predicted_labels, all_probs_positive = [], [], []
    print("\n--- Evaluating on Test Set ---")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set", file=sys.stdout):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_batch = torch.max(outputs, 1)
            probs_batch = torch.softmax(outputs, dim=1)[:, 1]
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted_batch.cpu().numpy())
            all_probs_positive.extend(probs_batch.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels, pos_label=1, zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, pos_label=1, zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(all_true_labels, all_probs_positive)
    except ValueError:
        auc = float('nan'); print("AUC calculation failed (likely only one class in predictions/labels).")
    cm = confusion_matrix(all_true_labels, all_predicted_labels)

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (REAL): {precision:.4f}")
    print(f"Recall (REAL): {recall:.4f}")
    print(f"F1-Score (REAL): {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    plt.figure(figsize=(8, 6))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE (0)', 'REAL (1)'],
               yticklabels=['FAKE (0)', 'REAL (1)'])
    plt.xlabel('Predicted Label');
    plt.ylabel('True Label');
    plt.title('Confusion Matrix - Test Set')

    model_dir = os.path.dirname(TRAINED_MODEL_PATH) if os.path.dirname(TRAINED_MODEL_PATH) else '.'
    cm_save_path = os.path.join(model_dir, f'confusion_matrix_test_local_seq{SEQUENCE_LENGTH}.png')
    plt.savefig(cm_save_path);
    print(f"Confusion matrix plot: {cm_save_path}");
    plt.show()


if __name__ == '__main__':
    main()