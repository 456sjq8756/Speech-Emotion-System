import os
import librosa
import soundfile as sf

dataset_path = 'F:\PythonProject1\Speech-Emotion-Recognition-master\pytorch_ver\dataset_union' # 填入你合并后的根目录

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            # 加载并重采样
            data, sr = librosa.load(file_path, sr=16000)
            # 覆盖原文件
            sf.write(file_path, data, 16000)
print("所有音频已统一转换为 16000Hz")
