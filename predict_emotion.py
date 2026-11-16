#!/usr/bin/env python3
"""
CLI Tool: predict_emotion.py
============================
Performs Multimodal Audio-Visual Emotion Recognition on synchronized video
and vocal audio recordings, comparing unimodal and multimodal predictions.

Usage Examples:
  # Multimodal Mode (Video + Audio)
  python predict_emotion.py --video outputs/sample_happy.mp4 --audio outputs/sample_happy.wav

  # Audio-Only Mode
  python predict_emotion.py --audio outputs/sample_angry.wav

  # Visual-Only Mode
  python predict_emotion.py --video outputs/sample_sad.mp4
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from main import (
    AudioVisualEmotionNet,
    EMOTIONS,
    NUM_EMOTIONS,
    SAMPLE_RATE,
    AUDIO_DURATION,
    NUM_VIDEO_FRAMES,
    IMG_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_prob_bar(prob, length=22):
    filled = int(round(prob * length))
    bar = "█" * filled + "░" * (length - filled)
    return bar


def load_model(checkpoint_path, device=DEVICE):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    emotions = checkpoint.get("emotions", EMOTIONS)
    d_model = checkpoint.get("d_model", 128)

    model = AudioVisualEmotionNet(
        num_classes=len(emotions),
        d_model=d_model,
        nhead=4
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, emotions


def preprocess_video(video_path, target_frames=NUM_VIDEO_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
        raw_frames.append(resized)
    cap.release()

    if not raw_frames:
        raise ValueError(f"No frames found in video: {video_path}")

    if len(raw_frames) == target_frames:
        sampled = raw_frames
    elif len(raw_frames) > target_frames:
        indices = np.linspace(0, len(raw_frames) - 1, target_frames, dtype=int)
        sampled = [raw_frames[i] for i in indices]
    else:
        sampled = list(raw_frames)
        while len(sampled) < target_frames:
            sampled.append(raw_frames[-1])

    arr = np.stack(sampled, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
    return tensor


def preprocess_audio(audio_path, target_sr=SAMPLE_RATE, target_dur=AUDIO_DURATION):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        import scipy.signal as signal
        num_s = int(len(data) * target_sr / sr)
        data = signal.resample(data, num_s).astype(np.float32)

    req_samples = int(target_sr * target_dur)
    if len(data) < req_samples:
        padded = np.zeros(req_samples, dtype=np.float32)
        padded[:len(data)] = data
        data = padded
    else:
        data = data[:req_samples]

    max_v = np.max(np.abs(data))
    if max_v > 1e-4:
        data = data / max_v * 0.90

    wave = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        n_mels=128,
        f_min=20.0,
        f_max=target_sr // 2,
        power=2.0
    )
    mel = torch.log(torch.clamp(mel_transform(wave), min=1e-5))
    return mel


def run_emotion_prediction(video_path=None, audio_path=None,
                           checkpoint_path="models/multimodal_emotion_net_best.pth"):
    t_start = time.perf_counter()

    print("\n" + "=" * 78)
    print(" AUDIO-VISUAL EMOTION RECOGNITION (MULTIMODAL)")
    print("=" * 78)

    has_video = video_path is not None and os.path.exists(video_path)
    has_audio = audio_path is not None and os.path.exists(audio_path)

    if not has_video and not has_audio:
        print("Error: Please provide at least one modality (--video and/or --audio).")
        sys.exit(1)

    # Determine mode
    if has_video and has_audio:
        mode_str = "MULTIMODAL (Audio + Video)"
    elif has_audio:
        mode_str = "UNIMODAL AUDIO-ONLY"
    else:
        mode_str = "UNIMODAL VISUAL-ONLY"

    print(f"Operational Mode:  {mode_str}")
    if has_video:
        print(f"Input Face Video:  {os.path.basename(video_path)} ({video_path})")
    if has_audio:
        print(f"Input Vocal Audio: {os.path.basename(audio_path)} ({audio_path})")
    print(f"Inference Device:  {DEVICE}")

    # Load Model
    model, emotions = load_model(checkpoint_path, device=DEVICE)

    # Preprocessing
    t_prep_0 = time.perf_counter()
    v_tensor = preprocess_video(video_path).to(DEVICE) if has_video else None
    m_tensor = preprocess_audio(audio_path).to(DEVICE) if has_audio else None
    t_prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

    # Model Forward Pass
    t_infer_0 = time.perf_counter()
    with torch.no_grad():
        if has_video and has_audio:
            l_fused, l_aud, l_vis, _ = model(m_tensor, v_tensor)
            p_fused = F.softmax(l_fused, dim=-1)[0].cpu().numpy()
            p_aud = F.softmax(l_aud, dim=-1)[0].cpu().numpy()
            p_vis = F.softmax(l_vis, dim=-1)[0].cpu().numpy()
        elif has_audio:
            l_aud = model.forward_audio_only(m_tensor)
            p_aud = F.softmax(l_aud, dim=-1)[0].cpu().numpy()
            p_fused = p_aud
            p_vis = None
        else: # has_video only
            l_vis = model.forward_visual_only(v_tensor)
            p_vis = F.softmax(l_vis, dim=-1)[0].cpu().numpy()
            p_fused = p_vis
            p_aud = None
    t_infer_ms = (time.perf_counter() - t_infer_0) * 1000.0

    # Format Results
    best_idx = np.argmax(p_fused)
    best_emotion = emotions[best_idx]
    best_conf = p_fused[best_idx] * 100.0

    print("\n" + "-" * 78)
    if has_video and has_audio:
        print(" PREDICTED EMOTION PROBABILITIES ACROSS MODALITIES")
        print("-" * 78)
        print(f" {'Emotion':<12}  {'Multimodal':<12}  {'Audio-Only':<12}  {'Visual-Only':<12}  {'Confidence Level'}")
        print(" " + "-" * 74)
        for i, emo in enumerate(emotions):
            bar = format_prob_bar(p_fused[i], length=18)
            marker = "★ [PREDICTED]" if i == best_idx else ""
            print(f"  {emo:<12}  {p_fused[i]*100:>6.2f}%       {p_aud[i]*100:>6.2f}%       {p_vis[i]*100:>6.2f}%       {bar}  {marker}")
    else:
        print(" PREDICTED EMOTION PROBABILITIES")
        print("-" * 78)
        print(f" {'Rank':<4}  {'Emotion':<14}  {'Probability':<12}  {'Confidence Level'}")
        print(" " + "-" * 74)
        ranked = np.argsort(-p_fused)
        for r, i in enumerate(ranked, 1):
            bar = format_prob_bar(p_fused[i], length=24)
            marker = "★ [PREDICTED]" if r == 1 else ""
            print(f"  {r:<3}  {emotions[i]:<14}  {p_fused[i]*100:>6.2f}%       {bar}  {marker}")

    print("\n" + "=" * 78)
    print(f" >>> FINAL PREDICTED EMOTION:  \"{best_emotion}\"  ({best_conf:.2f}% Confidence)")
    print("=" * 78)

    t_total_ms = (time.perf_counter() - t_start) * 1000.0
    rtf = t_total_ms / (AUDIO_DURATION * 1000.0)
    speedup = (AUDIO_DURATION * 1000.0) / t_total_ms

    print("\n" + "-" * 78)
    print(" INFERENCE PERFORMANCE & REAL-TIME FACTOR")
    print("-" * 78)
    print(f"  • Preprocessing Latency:  {t_prep_ms:6.2f} ms")
    print(f"  • Neural Inference Time:  {t_infer_ms:6.2f} ms")
    print(f"  • Total Pipeline Latency: {t_total_ms:6.2f} ms")
    print(f"  • Real-Time Factor (RTF): {rtf:6.4f} ({speedup:.1f}x real-time speedup)")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Audio-Visual Multimodal Emotion Recognition (Face Video + Vocal Audio)."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="/home/sericsoft/Desktop/pytorch_100_projects/96_multimodal_audio_visual_emotion_recognition/outputs/sample_happy.mp4",
        help="Path to input facial video (.mp4)."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="/home/sericsoft/Desktop/pytorch_100_projects/96_multimodal_audio_visual_emotion_recognition/outputs/sample_happy.wav",
        help="Path to input vocal audio (.wav)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/sericsoft/Desktop/pytorch_100_projects/96_multimodal_audio_visual_emotion_recognition/models/multimodal_emotion_net_best.pth",
        help="Path to trained model checkpoint."
    )

    args = parser.parse_args()
    run_emotion_prediction(
        video_path=args.video,
        audio_path=args.audio,
        checkpoint_path=args.checkpoint
    )


if __name__ == "__main__":
    main()

