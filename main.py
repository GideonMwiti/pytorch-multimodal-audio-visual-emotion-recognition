#!/usr/bin/env python3
"""
Project 96: Audio-Visual Emotion Recognition (Multimodal)
=========================================================
Combines facial visual expressions with vocal acoustic prosody to classify
human emotional states (HAPPY, SAD, ANGRY, SURPRISED, NEUTRAL, FEARFUL).

Architecture:
  1. Audio Branch: 128-band Log-Mel Spectrogram frontend + 2D CNN-SE ResNet.
  2. Visual Branch: 3D-CNN spatiotemporal backbone over 16 face video frames.
  3. Cross-Modal Bidirectional Attention: Mutual attention between acoustic
     temporal frames and visual spatiotemporal features.
  4. Multimodal Fusion Head: Concatenates pooled unimodal embeddings with
     cross-attended multimodal context vectors for classification.
  5. Multi-Branch Benchmarking: Explicitly measures Audio-Only, Visual-Only,
     and Multimodal Fusion performance to demonstrate cross-modal superiority.
"""

import os
import math
import time
import json
import random
from collections import Counter

import numpy as np
import cv2
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Fix random seeds for reproducible experiments
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion Categories
EMOTIONS = ["HAPPY", "SAD", "ANGRY", "SURPRISED", "NEUTRAL", "FEARFUL"]
NUM_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

SAMPLE_RATE = 16000
AUDIO_DURATION = 2.0  # seconds
NUM_AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

NUM_VIDEO_FRAMES = 16 # 16 frames @ 8 fps
IMG_SIZE = 64         # 64x64 face ROI


# ============================================================================
# 1. Procedural Audio-Visual Multimodal Synthesizer
# ============================================================================

class MultimodalEmotionSynthesizer:
    """Synthesizes synchronized facial video sequences and emotional speech audio."""

    @staticmethod
    def render_face_frame(emotion, progress, base_skin=185, dx=0, dy=0, img_size=IMG_SIZE):
        """
        Renders a 64x64 face frame exhibiting psychological Action Units (AUs)
        intensifying along the temporal progress trajectory (0.0 to 1.0).
        """
        frame = np.full((img_size, img_size), 220, dtype=np.uint8) # background
        cx = int(img_size // 2 + dx)
        cy = int(img_size // 2 + dy)
        intensity = float(0.3 + 0.7 * progress)

        # 1. Head Oval
        cv2.ellipse(frame, (cx, cy), (22, 27), 0, 0, 360, base_skin, -1)

        # 2. Eyes (Sclera & Pupils)
        eye_h = 4
        if emotion in ["SURPRISED", "FEARFUL"]:
            eye_h = int(4 + 4.5 * intensity) # AU5: Wide open
        elif emotion in ["ANGRY", "HAPPY"]:
            eye_h = max(2, int(4 - 1.5 * intensity)) # AU6/AU7: Squint / cheek raise
        elif emotion == "SAD":
            eye_h = max(2, int(4 - 1.0 * intensity))

        # Left & Right eyes
        cv2.ellipse(frame, (cx - 9, cy - 6), (5, eye_h), 0, 0, 360, 255, -1)
        cv2.ellipse(frame, (cx + 9, cy - 6), (5, eye_h), 0, 0, 360, 255, -1)
        # Pupils
        cv2.circle(frame, (cx - 9, cy - 6), 2, 40, -1)
        cv2.circle(frame, (cx + 9, cy - 6), 2, 40, -1)

        # 3. Eyebrows
        if emotion == "ANGRY":
            # AU4: Brow lowerer / furrow (V-shape)
            cv2.line(frame, (cx - 14, cy - 13), (cx - 4, int(cy - 8 + 3.5 * intensity)), 45, 2)
            cv2.line(frame, (cx + 14, cy - 13), (cx + 4, int(cy - 8 + 3.5 * intensity)), 45, 2)
        elif emotion in ["SURPRISED", "FEARFUL"]:
            # AU1+AU2: Arched raised eyebrows
            cv2.ellipse(frame, (cx - 9, int(cy - 14 - 3.5 * intensity)), (6, 3), 0, 180, 360, 45, 2)
            cv2.ellipse(frame, (cx + 9, int(cy - 14 - 3.5 * intensity)), (6, 3), 0, 180, 360, 45, 2)
        elif emotion == "SAD":
            # AU1: Inner brow raiser (inverted V)
            cv2.line(frame, (cx - 14, int(cy - 9 + 2.5 * intensity)), (cx - 4, cy - 14), 45, 2)
            cv2.line(frame, (cx + 14, int(cy - 9 + 2.5 * intensity)), (cx + 4, cy - 14), 45, 2)
        else: # HAPPY, NEUTRAL
            cv2.line(frame, (cx - 14, cy - 12), (cx - 4, cy - 12), 45, 2)
            cv2.line(frame, (cx + 14, cy - 12), (cx + 4, cy - 12), 45, 2)

        # 4. Mouth & Oral Expression
        if emotion == "HAPPY":
            # AU12: Lip corner puller (upturned smile curve)
            w_m = int(8 + 5 * intensity)
            h_m = int(4 + 5 * intensity)
            cv2.ellipse(frame, (cx, cy + 12), (w_m, h_m), 0, 0, 180, 40, 2)
            # Teeth visible in big smile
            if intensity > 0.6:
                cv2.rectangle(frame, (cx - w_m // 2, cy + 12), (cx + w_m // 2, cy + 14), 245, -1)
        elif emotion == "SAD":
            # AU15: Lip corner depressor (drooping frown curve)
            w_m = int(7 + 3 * intensity)
            h_m = int(3 + 4 * intensity)
            cv2.ellipse(frame, (cx, cy + 18), (w_m, h_m), 0, 180, 360, 40, 2)
        elif emotion == "SURPRISED":
            # AU26/AU27: Jaw drop (vertical open oral aperture)
            w_m = 5
            h_m = int(5 + 6 * intensity)
            cv2.ellipse(frame, (cx, cy + 14), (w_m, h_m), 0, 0, 360, 25, -1)
        elif emotion == "ANGRY":
            # AU23/AU24: Tightly compressed lips
            w_m = int(8 + 2 * intensity)
            cv2.line(frame, (cx - w_m, cy + 14), (cx + w_m, cy + 14), 35, 2)
        elif emotion == "FEARFUL":
            # AU20: Horizontal lip stretcher
            w_m = int(10 + 4 * intensity)
            cv2.ellipse(frame, (cx, cy + 14), (w_m, 4), 0, 0, 360, 35, -1)
        else: # NEUTRAL
            cv2.line(frame, (cx - 7, cy + 14), (cx + 7, cy + 14), 60, 2)

        # Add subtle skin texture noise
        noise = np.random.normal(0, 2.0, (img_size, img_size)).astype(np.float32)
        final_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return final_frame

    @staticmethod
    def synthesize_video(emotion, num_frames=NUM_VIDEO_FRAMES, augment=True):
        """Generates 16-frame face video with temporal expression progression."""
        frames = []
        base_skin = random.randint(170, 205) if augment else 185
        dx = random.randint(-2, 2) if augment else 0
        dy = random.randint(-2, 2) if augment else 0

        for f_idx in range(num_frames):
            progress = f_idx / (num_frames - 1)
            frame = MultimodalEmotionSynthesizer.render_face_frame(
                emotion=emotion,
                progress=progress,
                base_skin=base_skin,
                dx=dx,
                dy=dy
            )
            frames.append(frame)

        # Shape: [T, H, W] normalized [0, 1]
        video_arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return video_arr

    @staticmethod
    def synthesize_audio(emotion, sr=SAMPLE_RATE, dur=AUDIO_DURATION, augment=True):
        """
        Synthesizes 2.0s emotional vocal prosody with characteristic pitch contours,
        vibrato, harmonic energy, and envelope dynamics.
        """
        N = int(sr * dur)
        t = np.linspace(0, dur, N, endpoint=False)
        pitch_jitter = np.random.uniform(-10.0, 10.0) if augment else 0.0

        if emotion == "HAPPY":
            f0 = 270.0 + pitch_jitter + 60.0 * np.sin(2 * np.pi * 2.0 * t)
            env = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
            harmonics = [1.0, 0.70, 0.45, 0.30]
        elif emotion == "SAD":
            f0 = 130.0 + pitch_jitter - 25.0 * (t / dur)
            env = 0.4 + 0.3 * np.exp(-t * 0.8)
            harmonics = [1.0, 0.25, 0.08, 0.02]
        elif emotion == "ANGRY":
            f0 = 340.0 + pitch_jitter + 50.0 * np.sin(2 * np.pi * 4.0 * t)
            env = 0.8 + 0.2 * np.sin(2 * np.pi * 5.0 * t) ** 2
            harmonics = [1.0, 0.90, 0.70, 0.55]
        elif emotion == "SURPRISED":
            f0 = 190.0 + pitch_jitter + 260.0 * (1.0 / (1.0 + np.exp(-(t - 0.6) * 12.0)))
            env = 0.4 + 0.5 * np.exp(-((t - 0.7) ** 2) / 0.25)
            harmonics = [1.0, 0.60, 0.30, 0.15]
        elif emotion == "FEARFUL":
            f0 = 230.0 + pitch_jitter + 40.0 * np.sin(2 * np.pi * 8.5 * t)
            env = 0.5 + 0.35 * np.sin(2 * np.pi * 8.0 * t)
            harmonics = [1.0, 0.50, 0.35, 0.20]
        else: # NEUTRAL
            f0 = 165.0 + pitch_jitter + 8.0 * np.sin(2 * np.pi * 0.5 * t)
            env = np.ones(N) * 0.65
            harmonics = [1.0, 0.40, 0.20, 0.08]

        phase = 2 * np.pi * np.cumsum(f0) / sr
        sig = np.zeros(N)
        for h_idx, h_weight in enumerate(harmonics, 1):
            sig += h_weight * np.sin(h_idx * phase)
        sig = sig * env

        # Add consistent room noise
        sig += np.random.randn(N) * 0.01

        # Peak normalization
        max_v = np.max(np.abs(sig))
        if max_v > 1e-4:
            sig = sig / max_v * 0.88
        return sig.astype(np.float32)


# ============================================================================
# 2. Multimodal Neural Network Architecture
# ============================================================================

class SqueezeExcitation2D(nn.Module):
    """SE Attention for 2D Audio Feature Maps."""
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 8)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class AudioEmotionEncoder(nn.Module):
    """Encodes Log-Mel Spectrogram into temporal acoustic tokens and pooled vectors."""
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SqueezeExcitation2D(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, d_model, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            SqueezeExcitation2D(d_model)
        )
        # Collapse remaining 16 mel bins to 1
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, mel):
        # mel: [B, 1, 128, T_mel]
        x = self.conv1(mel)  # [B, 32, 64, T/2]
        x = self.conv2(x)    # [B, 64, 32, T/4]
        x = self.conv3(x)    # [B, d_model, 16, T/4]
        x = self.freq_pool(x).squeeze(2) # [B, d_model, T']
        tokens = x.transpose(1, 2)       # [B, T', d_model]

        # Dual Global Pooling (GAP + GMP)
        gap = tokens.mean(dim=1)
        gmp = tokens.amax(dim=1)
        pooled = self.proj(torch.cat([gap, gmp], dim=1)) # [B, d_model]
        return tokens, pooled


class VisualEmotionEncoder(nn.Module):
    """Encodes silent facial video into spatiotemporal tokens and pooled vectors."""
    def __init__(self, d_model=128):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(24),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), # 16x16
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.Conv3d(48, d_model, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)) # Collapse spatial dimensions
        )
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, video):
        # video: [B, 1, T_vis, 64, 64]
        x = self.frontend(video).squeeze(-1).squeeze(-1) # [B, d_model, T']
        tokens = x.transpose(1, 2)                       # [B, T', d_model]

        gap = tokens.mean(dim=1)
        gmp = tokens.amax(dim=1)
        pooled = self.proj(torch.cat([gap, gmp], dim=1)) # [B, d_model]
        return tokens, pooled


class CrossModalBidirectionalAttention(nn.Module):
    """Bidirectional Multihead Cross-Attention between acoustic and visual streams."""
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.mha_a2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_v2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

    def forward(self, h_aud, h_vis):
        # Audio attends to Visual
        a2v, w_a2v = self.mha_a2v(h_aud, h_vis, h_vis)
        out_a = self.norm_a(h_aud + a2v)

        # Visual attends to Audio
        v2a, w_v2a = self.mha_v2a(h_vis, h_aud, h_aud)
        out_v = self.norm_v(h_vis + v2a)
        return out_a, out_v, w_a2v, w_v2a


class AudioVisualEmotionNet(nn.Module):
    """
    Multimodal Audio-Visual Emotion Recognition Network.
    Integrates Audio ResNet-SE, Visual 3D-CNN, Bidirectional Cross-Modal Attention,
    and unimodal baseline heads for comparative benchmarking.
    """
    def __init__(self, num_classes=NUM_EMOTIONS, d_model=128, nhead=4, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.audio_encoder = AudioEmotionEncoder(d_model=d_model)
        self.visual_encoder = VisualEmotionEncoder(d_model=d_model)
        self.cross_attn = CrossModalBidirectionalAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        # Unimodal Auxiliary Heads (for ablation benchmarking)
        self.audio_head = nn.Linear(d_model, num_classes)
        self.visual_head = nn.Linear(d_model, num_classes)

        # Multimodal Fusion Head: Concatenates (z_aud, z_vis, c_aud, c_vis) -> 4 * d_model
        self.multimodal_head = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, video):
        # 1. Unimodal Encoders
        h_aud, z_aud = self.audio_encoder(mel)     # [B, T_a, D], [B, D]
        h_vis, z_vis = self.visual_encoder(video)  # [B, T_v, D], [B, D]

        # 2. Cross-Modal Attention
        c_aud_seq, c_vis_seq, w_a2v, w_v2a = self.cross_attn(h_aud, h_vis)
        c_aud = c_aud_seq.mean(dim=1) # [B, D]
        c_vis = c_vis_seq.mean(dim=1) # [B, D]

        # 3. Multimodal Representation Fusion
        fused = torch.cat([z_aud, z_vis, c_aud, c_vis], dim=1) # [B, 4 * D]

        # 4. Predictions
        logits_fused = self.multimodal_head(fused)
        logits_aud = self.audio_head(z_aud)
        logits_vis = self.visual_head(z_vis)
        return logits_fused, logits_aud, logits_vis, w_a2v

    def forward_audio_only(self, mel):
        _, z_aud = self.audio_encoder(mel)
        return self.audio_head(z_aud)

    def forward_visual_only(self, video):
        _, z_vis = self.visual_encoder(video)
        return self.visual_head(z_vis)


# ============================================================================
# 3. Dataset & DataLoaders
# ============================================================================

class MultimodalEmotionDataset(Dataset):
    """Dataset serving synchronized audio waveforms, video arrays, and emotion labels."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wave = torch.from_numpy(item["audio"]).unsqueeze(0).float() # [1, num_samples]
        vid = torch.from_numpy(item["video"]).unsqueeze(0).float()  # [1, T, H, W]
        label = torch.tensor(item["label"], dtype=torch.long)
        return wave, vid, label


class MultimodalBatchCollate:
    """Computes Log-Mel Spectrogram dynamically on audio batch."""
    def __init__(self, n_mels=128, n_fft=1024, hop_length=512):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=SAMPLE_RATE // 2,
            power=2.0
        )

    def __call__(self, batch):
        waves, vids, labels = zip(*batch)
        waves = torch.stack(waves, dim=0)   # [B, 1, num_samples]
        vids = torch.stack(vids, dim=0)     # [B, 1, T, H, W]
        labels = torch.stack(labels, dim=0) # [B]

        mels = self.mel_transform(waves)
        log_mels = torch.log(torch.clamp(mels, min=1e-5))
        return log_mels, vids, labels


def generate_multimodal_dataset(samples_per_emotion=100, is_train=True):
    """Generates balanced synthetic audio-visual emotion dataset."""
    mode = "train" if is_train else "validation"
    print(f"Generating {mode} multimodal emotion dataset ({samples_per_emotion * NUM_EMOTIONS} pairs)...")
    samples = []
    synth = MultimodalEmotionSynthesizer()

    for emotion_idx, emotion in enumerate(EMOTIONS):
        for _ in range(samples_per_emotion):
            video = synth.synthesize_video(emotion, num_frames=NUM_VIDEO_FRAMES, augment=is_train)
            audio = synth.synthesize_audio(emotion, sr=SAMPLE_RATE, dur=AUDIO_DURATION, augment=is_train)
            samples.append({
                "audio": audio,
                "video": video,
                "emotion": emotion,
                "label": emotion_idx
            })
    random.shuffle(samples)
    return samples


# ============================================================================
# 4. Evaluation Suite: Multimodal vs Audio-Only vs Visual-Only
# ============================================================================

def evaluate_multimodal_network(model, loader, device=DEVICE):
    """
    Evaluates Multimodal Fusion, Audio-Only, and Visual-Only classification
    accuracy, macro F1, and confusion matrices.
    """
    model.eval()
    all_targets = []
    preds_fused = []
    preds_aud = []
    preds_vis = []
    probs_fused = []

    with torch.no_grad():
        for mels, vids, labels in loader:
            mels = mels.to(device)
            vids = vids.to(device)
            labels = labels.to(device)

            l_fused, l_aud, l_vis, _ = model(mels, vids)

            all_targets.extend(labels.cpu().numpy())
            preds_fused.extend(torch.argmax(l_fused, dim=-1).cpu().numpy())
            preds_aud.extend(torch.argmax(l_aud, dim=-1).cpu().numpy())
            preds_vis.extend(torch.argmax(l_vis, dim=-1).cpu().numpy())
            probs_fused.append(F.softmax(l_fused, dim=-1).cpu().numpy())

    all_targets = np.array(all_targets)
    preds_fused = np.array(preds_fused)
    preds_aud = np.array(preds_aud)
    preds_vis = np.array(preds_vis)
    probs_fused = np.concatenate(probs_fused, axis=0)

    acc_fused = float(np.mean(preds_fused == all_targets))
    acc_aud = float(np.mean(preds_aud == all_targets))
    acc_vis = float(np.mean(preds_vis == all_targets))

    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        all_targets, preds_fused, average="macro", zero_division=0
    )
    prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
        all_targets, preds_aud, average="macro", zero_division=0
    )
    prec_v, rec_v, f1_v, _ = precision_recall_fscore_support(
        all_targets, preds_vis, average="macro", zero_division=0
    )

    conf_mat = confusion_matrix(all_targets, preds_fused, labels=range(NUM_EMOTIONS))

    return {
        "multimodal_accuracy": acc_fused,
        "multimodal_macro_f1": float(f1_f),
        "multimodal_precision": float(prec_f),
        "multimodal_recall": float(rec_f),
        "audio_only_accuracy": acc_aud,
        "audio_only_macro_f1": float(f1_a),
        "visual_only_accuracy": acc_vis,
        "visual_only_macro_f1": float(f1_v),
        "confusion_matrix": conf_mat.tolist()
    }


# ============================================================================
# 5. Training Pipeline
# ============================================================================

def train_audio_visual_emotion_system():
    """Trains the full Multimodal Audio-Visual Emotion Recognition system."""
    print("=" * 78)
    print(" PROJECT 96: AUDIO-VISUAL EMOTION RECOGNITION (MULTIMODAL)")
    print("=" * 78)

    output_dir = "/home/sericsoft/Desktop/pytorch_100_projects/96_multimodal_audio_visual_emotion_recognition/outputs"
    model_dir = "/home/sericsoft/Desktop/pytorch_100_projects/96_multimodal_audio_visual_emotion_recognition/models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Dataset Generation
    train_data = generate_multimodal_dataset(samples_per_emotion=100, is_train=True) # 600 pairs
    val_data = generate_multimodal_dataset(samples_per_emotion=30, is_train=False)   # 180 pairs

    collate_fn = MultimodalBatchCollate(n_mels=128)
    train_loader = DataLoader(MultimodalEmotionDataset(train_data), batch_size=20, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MultimodalEmotionDataset(val_data), batch_size=20, shuffle=False, collate_fn=collate_fn)

    # 2. Model Initialization
    model = AudioVisualEmotionNet(
        num_classes=NUM_EMOTIONS,
        d_model=128,
        nhead=4,
        dropout=0.15
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized AudioVisualEmotionNet on {DEVICE} ({total_params:,} trainable parameters).")

    # 3. Loss & Optimizer
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=6e-5)

    history = {
        "train_loss": [],
        "val_fused_acc": [],
        "val_aud_acc": [],
        "val_vis_acc": []
    }

    best_acc = -1.0
    best_model_path = os.path.join(model_dir, "multimodal_emotion_net_best.pth")

    print("\nStarting Multimodal Audio-Visual Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for mels, vids, labels in train_loader:
            mels = mels.to(DEVICE)
            vids = vids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            l_fused, l_aud, l_vis, _ = model(mels, vids)

            # Joint multi-task loss
            loss_fused = ce_loss_fn(l_fused, labels)
            loss_aud = ce_loss_fn(l_aud, labels)
            loss_vis = ce_loss_fn(l_vis, labels)
            loss = loss_fused + 1.0 * loss_aud + 1.0 * loss_vis

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_multimodal_network(model, val_loader, device=DEVICE)
        fused_acc = val_metrics["multimodal_accuracy"]
        aud_acc = val_metrics["audio_only_accuracy"]
        vis_acc = val_metrics["visual_only_accuracy"]

        history["train_loss"].append(avg_train_loss)
        history["val_fused_acc"].append(fused_acc)
        history["val_aud_acc"].append(aud_acc)
        history["val_vis_acc"].append(vis_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | "
              f"Multimodal Acc: {fused_acc*100:.2f}% | Audio Acc: {aud_acc*100:.2f}% | Visual Acc: {vis_acc*100:.2f}%")

        if fused_acc > best_acc:
            best_acc = fused_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "emotions": EMOTIONS,
                "d_model": 128,
                "epoch": epoch,
                "val_accuracy": fused_acc
            }, best_model_path)

    print(f"\nModel training completed! Best checkpoint saved to: {best_model_path}")

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate_multimodal_network(model, val_loader, device=DEVICE)

    # 4. Latency Benchmark
    print("\nRunning Inference Latency Benchmark on CPU...")
    test_wave = torch.from_numpy(val_data[0]["audio"]).unsqueeze(0).unsqueeze(0).float()
    test_vid = torch.from_numpy(val_data[0]["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    mel_ext = collate_fn.mel_transform

    latencies_fused = []
    latencies_aud = []
    latencies_vis = []

    for _ in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            t_mel = torch.log(torch.clamp(mel_ext(test_wave), min=1e-5)).to(DEVICE)
            _ = model(t_mel, test_vid)
        latencies_fused.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.forward_audio_only(t_mel)
        latencies_aud.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.forward_visual_only(test_vid)
        latencies_vis.append((time.perf_counter() - t0) * 1000.0)

    avg_fused_ms = float(np.mean(latencies_fused[5:]))
    avg_aud_ms = float(np.mean(latencies_aud[5:]))
    avg_vis_ms = float(np.mean(latencies_vis[5:]))

    final_metrics["latency_multimodal_ms"] = round(avg_fused_ms, 2)
    final_metrics["latency_audio_only_ms"] = round(avg_aud_ms, 2)
    final_metrics["latency_visual_only_ms"] = round(avg_vis_ms, 2)
    final_metrics["speedup_realtime"] = round((AUDIO_DURATION * 1000.0) / avg_fused_ms, 1)
    final_metrics["rtf"] = round(avg_fused_ms / (AUDIO_DURATION * 1000.0), 4)

    print(f"\nFinal Comparative Benchmark Results:")
    print(f"  ★ Multimodal Fusion Accuracy:  {final_metrics['multimodal_accuracy']*100:.2f}% (Macro F1: {final_metrics['multimodal_macro_f1']*100:.2f}%)")
    print(f"  • Audio-Only Accuracy:          {final_metrics['audio_only_accuracy']*100:.2f}% (Macro F1: {final_metrics['audio_only_macro_f1']*100:.2f}%)")
    print(f"  • Visual-Only Accuracy:         {final_metrics['visual_only_accuracy']*100:.2f}% (Macro F1: {final_metrics['visual_only_macro_f1']*100:.2f}%)")
    print(f"  • Multimodal Latency:           {final_metrics['latency_multimodal_ms']} ms ({final_metrics['speedup_realtime']}x real-time speedup)")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "multimodal_emotion_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # 5. Export Canonical Test Audio-Video Files
    print("\nExporting Canonical Multimodal Emotion Video and WAV Pairs...")
    synth = MultimodalEmotionSynthesizer()
    saved_samples = []

    for emotion in EMOTIONS:
        vid = synth.synthesize_video(emotion, num_frames=NUM_VIDEO_FRAMES, augment=False)
        aud = synth.synthesize_audio(emotion, sr=SAMPLE_RATE, dur=AUDIO_DURATION, augment=False)

        vid_filename = f"sample_{emotion.lower()}.mp4"
        aud_filename = f"sample_{emotion.lower()}.wav"
        vid_path = os.path.join(output_dir, vid_filename)
        aud_path = os.path.join(output_dir, aud_filename)

        # Write WAV
        sf.write(aud_path, aud, SAMPLE_RATE)

        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(vid_path, fourcc, 8.0, (IMG_SIZE, IMG_SIZE), isColor=False)
        for frame_idx in range(NUM_VIDEO_FRAMES):
            frame_uint8 = (vid[frame_idx] * 255.0).astype(np.uint8)
            out_writer.write(frame_uint8)
        out_writer.release()

        saved_samples.append({
            "emotion": emotion,
            "video": vid,
            "audio": aud,
            "vid_path": vid_path,
            "aud_path": aud_path,
            "vid_filename": vid_filename,
            "aud_filename": aud_filename
        })
        print(f"  Saved: {vid_filename} & {aud_filename} (Emotion: '{emotion}')")

    # 6. Generate Diagnostic Dashboard
    print("\nGenerating Diagnostic Visualization Dashboard...")
    report_path = os.path.join(output_dir, "multimodal_emotion_report.png")
    generate_diagnostic_report(
        model=model,
        collate_fn=collate_fn,
        history=history,
        metrics=final_metrics,
        saved_samples=saved_samples,
        report_path=report_path
    )
    print(f"Dashboard saved to: {report_path}")

    print("\n" + "=" * 78)
    print(" PROJECT 96 TRAINING AND BENCHMARKING COMPLETE!")
    print("=" * 78)


def generate_diagnostic_report(model, collate_fn, history, metrics, saved_samples, report_path):
    """Creates a publication-ready 6-panel diagnostic visualization report."""
    fig = plt.figure(figsize=(20, 14), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Panel 1: Synchronized Multimodal Input (Visual Face Strip + Audio Log-Mel)
    ax1 = fig.add_subplot(2, 3, 1)
    happy_sample = saved_samples[0] # HAPPY
    vid = happy_sample["video"]
    # 8 evenly spaced frames
    indices = np.linspace(0, NUM_VIDEO_FRAMES - 1, 8, dtype=int)
    strip = np.hstack([vid[i] for i in indices])
    ax1.imshow(strip, cmap="gray")
    ax1.set_title("1. Visual Face Progression: 'HAPPY' (8 of 16 Frames)", fontsize=11, fontweight="bold")
    ax1.axis("off")
    for pos, idx_f in enumerate(indices):
        ax1.text(pos * IMG_SIZE + IMG_SIZE // 2, IMG_SIZE + 8, f"F{idx_f+1}",
                 fontsize=8.5, ha="center")

    # 2. Panel 2: Comparative Modality Benchmark Bar Chart
    ax2 = fig.add_subplot(2, 3, 2)
    modalities = ["Audio-Only", "Visual-Only", "Multimodal Fusion"]
    accuracies = [
        metrics["audio_only_accuracy"] * 100,
        metrics["visual_only_accuracy"] * 100,
        metrics["multimodal_accuracy"] * 100
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax2.bar(modalities, accuracies, color=colors, edgecolor="black", width=0.5)
    for b, acc in zip(bars, accuracies):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{acc:.2f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax2.set_title("2. Modality Superiority Comparison", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # 3. Panel 3: Multimodal Confusion Matrix
    ax3 = fig.add_subplot(2, 3, 3)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype('float') / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
    im3 = ax3.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax3.set_xticks(range(NUM_EMOTIONS))
    ax3.set_yticks(range(NUM_EMOTIONS))
    ax3.set_xticklabels([e[:4] for e in EMOTIONS], fontsize=8.5)
    ax3.set_yticklabels(EMOTIONS, fontsize=8.5)
    ax3.set_xlabel("Predicted Emotion", fontsize=9.5)
    ax3.set_ylabel("True Emotion", fontsize=9.5)
    ax3.set_title(f"3. Multimodal Confusion Matrix ({metrics['multimodal_accuracy']*100:.1f}%)",
                  fontsize=11, fontweight="bold")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Panel 4: Cross-Modal Attention Alignment Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    w_t = torch.from_numpy(happy_sample["audio"]).unsqueeze(0).unsqueeze(0).float()
    v_t = torch.from_numpy(happy_sample["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    mel_ext = collate_fn.mel_transform
    m_t = torch.log(torch.clamp(mel_ext(w_t), min=1e-5)).to(DEVICE)
    with torch.no_grad():
        _, _, _, w_a2v = model(m_t, v_t)
    if w_a2v is not None:
        attn_mat = w_a2v[0].cpu().numpy()
        im4 = ax4.imshow(attn_mat, cmap="magma", aspect="auto", origin="lower")
        ax4.set_xlabel("Visual Temporal Frame Tokens (T_vis)", fontsize=9.5)
        ax4.set_ylabel("Acoustic Spectrogram Tokens (T_aud)", fontsize=9.5)
        ax4.set_title("4. Audio-to-Visual Cross-Modal Attention", fontsize=11, fontweight="bold")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Panel 5: Multi-Branch Training Convergence Curves
    ax5 = fig.add_subplot(2, 3, 5)
    ep_range = range(1, len(history["train_loss"]) + 1)
    ax5.plot(ep_range, [a * 100 for a in history["val_fused_acc"]], "o-", color="#2ca02c", linewidth=2.0, label="Multimodal Val Acc (%)")
    ax5.plot(ep_range, [a * 100 for a in history["val_aud_acc"]], "^--", color="#1f77b4", linewidth=1.5, label="Audio-Only Val Acc (%)")
    ax5.plot(ep_range, [a * 100 for a in history["val_vis_acc"]], "s:", color="#ff7f0e", linewidth=1.5, label="Visual-Only Val Acc (%)")
    ax5.set_xlabel("Epoch", fontsize=9.5)
    ax5.set_ylabel("Validation Accuracy (%)", fontsize=9.5)
    ax5.set_ylim(0, 105)
    ax5.set_title("5. Multi-Branch Convergence Dynamics", fontsize=11, fontweight="bold")
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(loc="lower right", fontsize=8.5)

    # 6. Panel 6: Showcase Emotion Test Predictions
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    title_text = "6. Multimodal Emotion Predictions Showcase\n"
    lines = [title_text]

    for item in saved_samples:
        w_item = torch.from_numpy(item["audio"]).unsqueeze(0).unsqueeze(0).float()
        v_item = torch.from_numpy(item["video"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        m_item = torch.log(torch.clamp(mel_ext(w_item), min=1e-5)).to(DEVICE)

        with torch.no_grad():
            lf, la, lv, _ = model(m_item, v_item)
            p_fused = F.softmax(lf, dim=-1)[0].cpu().numpy()
            p_aud = F.softmax(la, dim=-1)[0].cpu().numpy()
            p_vis = F.softmax(lv, dim=-1)[0].cpu().numpy()

            pred_f = EMOTIONS[np.argmax(p_fused)]
            conf_f = np.max(p_fused) * 100
            pred_a = EMOTIONS[np.argmax(p_aud)]
            pred_v = EMOTIONS[np.argmax(p_vis)]

        status = "✓ MATCH" if pred_f == item["emotion"] else "✗ ERROR"
        lines.append(f"• Target: '{item['emotion']}'")
        lines.append(f"  Multimodal: '{pred_f}' ({conf_f:.1f}%) [{status}] | Audio: '{pred_a}' | Visual: '{pred_v}'")
        lines.append("")

    lines.append(f"• CPU Pipeline Latency: {metrics['latency_multimodal_ms']} ms ({metrics['speedup_realtime']}x speedup)")

    ax6.text(0.02, 0.95, "\n".join(lines), transform=ax6.transAxes,
             fontsize=9.5, verticalalignment="top", fontfamily="sans-serif",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    fig.suptitle("Project 96: Multimodal Audio-Visual Emotion Recognition — Cross-Modal Attention & Benchmarks",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(report_path, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def main():
    train_audio_visual_emotion_system()


if __name__ == "__main__":
    main()
