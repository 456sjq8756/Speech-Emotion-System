import os
import librosa
import numpy as np
import tqdm
import pickle

# --- 配置参数 ---
DATA_PATH = "F:\PythonProject1\Speech-Emotion-Recognition-master\pytorch_ver\dataset_union"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40

# 情感标签映射
LABEL_MAP = {
    'angry': 0,
    'fear': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprise': 5
}


def get_features(y, sr):
    """
    输入音频数据 y，输出 MFCC 特征
    """
    try:
        # 1. 统一长度 (填充或截断)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 2. 提取 MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        return mfcc.T  # (Time, Features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def process_data():
    X = []
    y = []

    print("🚀 开始处理音频数据 (包含数据增强)...")

    # 遍历 data 文件夹
    for emotion_folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, emotion_folder)

        # 确定标签
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
                # 加载音频
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

                # --- 1. 原始数据 ---
                feat_original = get_features(audio, sr)
                if feat_original is not None:
                    X.append(feat_original)
                    y.append(label)

                # --- 2. 核心修改：变声数据增强 (对应隐私模式) ---
                # n_steps=4 必须与 app.py 里的参数一致！
                audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                feat_shifted = get_features(audio_shifted, sr)
                if feat_shifted is not None:
                    X.append(feat_shifted)
                    y.append(label)

            except Exception as e:
                print(f"文件出错 {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ 处理完成!")
    print(f"数据量翻倍: {X.shape[0]} 个样本 (原始+变声)")
    print(f"数据形状: {X.shape}")

    np.save("X.npy", X)
    np.save("y.npy", y)
    print("特征已保存为 X.npy 和 y.npy")


if __name__ == "__main__":
    process_data()
