import os
import librosa
import numpy as np
import tqdm

# --- 配置参数 ---
DATA_PATH = r"F:\PythonProject1\Speech-Emotion-Recognition-master\pytorch_ver\dataset_union"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40

# 情感标签映射
LABEL_MAP = {
    'angry': 0, 'fear': 1, 'happy': 2,
    'neutral': 3, 'sad': 4, 'surprise': 5
}


def get_features(y, sr):
    """提取 MFCC 特征"""
    try:
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return mfcc.T
    except Exception as e:
        print(f"提取特征出错: {e}")
        return None


def add_white_noise(data, noise_rate=0.015):
    """✨ 新增功能：添加背景白噪声，模拟复杂环境，提升鲁棒性"""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_rate * noise
    return augmented_data


def process_data():
    X, y = [], []
    print("🚀 开始处理音频数据 (包含变声与抗噪声数据增强)...")

    for emotion_folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, emotion_folder)
        label = -1
        for key, val in LABEL_MAP.items():
            if key in emotion_folder.lower():
                label = val
                break
        if label == -1 or not os.path.isdir(folder_path):
            continue

        print(f"正在处理类别: {emotion_folder} (Label: {label})")
        files = os.listdir(folder_path)

        for file_name in tqdm.tqdm(files):
            if not file_name.endswith('.wav'):
                continue
            file_path = os.path.join(folder_path, file_name)

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

                # 1. 原始数据
                feat_org = get_features(audio, sr)
                if feat_org is not None:
                    X.append(feat_org);
                    y.append(label)

                # 2. 变声脱敏数据 (原有的隐私保护)
                audio_shift = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                feat_shift = get_features(audio_shift, sr)
                if feat_shift is not None:
                    X.append(feat_shift);
                    y.append(label)

                # 3. ✨ 核心新增：噪声数据 (应对复杂环境)
                audio_noisy = add_white_noise(audio)
                feat_noisy = get_features(audio_noisy, sr)
                if feat_noisy is not None:
                    X.append(feat_noisy);
                    y.append(label)

            except Exception as e:
                pass

    X = np.array(X);
    y = np.array(y)
    print(f"\n✅ 处理完成! 数据量翻了三倍: 共 {X.shape[0]} 个样本 (原始+变声+噪声)")
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("特征已保存为 X.npy 和 y.npy")


if __name__ == "__main__":
    process_data()
