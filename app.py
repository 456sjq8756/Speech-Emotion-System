import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from model import CNN_LSTM_Model
import os

# --- 1. 配置参数 (必须与 train.py 一致) ---
INPUT_SIZE = 40
HIDDEN_SIZE = 128
NUM_CLASSES = 6
DURATION = 3
SAMPLE_RATE = 22050
DEVICE = torch.device('cpu')  # 推理时用 CPU 就够了

# 情感标签映射 (根据你 make_data.py 里的顺序，反向映射)
# 假设顺序是: 0:angry, 1:fear, 2:happy, 3:neutral, 4:sad, 5:surprise
# 请根据你实际训练时的 log 输出核对一下，如果不确定，先用这个试试
EMOTION_LABELS = {
    0: '愤怒 (Angry)',
    1: '恐惧 (Fear)',
    2: '快乐 (Happy)',
    3: '中性 (Neutral)',
    4: '悲伤 (Sad)',
    5: '惊讶 (Surprise)'
}


# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    model = CNN_LSTM_Model(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    # 加载你刚才训练好的权重
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    else:
        st.error("找不到 best_model.pth，请先运行 train.py！")
    model.eval()
    return model


# --- 3. 预处理函数 (逻辑必须与 make_data.py 一致) ---
def preprocess_audio(y, sr):
    # 统一长度
    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # 提取 MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=INPUT_SIZE)
    # 转置: (Features, Time) -> (Time, Features)
    mfcc = mfcc.T
    return mfcc


# --- 4. 页面布局 ---
st.set_page_config(page_title="语音情感识别系统", layout="wide")

st.title("🎙️ 基于 CNN-LSTM 的语音情感识别系统")
st.markdown("### 🎓 本科毕业设计演示 | 邵金桥")
st.write("---")

# 侧边栏
st.sidebar.header("功能控制")
privacy_mode = st.sidebar.checkbox("🛡️ 开启隐私保护模式 (变声脱敏)", value=False)
st.sidebar.info("说明：开启隐私模式后，系统将对音频进行变调处理，保护说话人音色，但模型仍能识别情感。")

# 主区域
col1, col2 = st.columns([1, 1])

uploaded_file = st.file_uploader("📂 请上传一段语音文件 (.wav)", type=['wav'])

if uploaded_file is not None:
    # 1. 加载音频
    y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, duration=DURATION)

    # 隐私保护处理 (变声)
    if privacy_mode:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)  # 升高4个半音
        st.toast("已应用隐私脱敏处理", icon="🛡️")

    # 2. 播放音频
    with col1:
        st.subheader("1. 音频播放 & 波形")
        # ✅ 修正后：直接播放处理后的信号 y
        # sample_rate 必须指定，否则播放速度会不对
        st.audio(y, sample_rate=sr)

        # 绘制波形图
        fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='blue')
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)

    # 3. 提取特征并推理
    mfcc_features = preprocess_audio(y, sr)

    # 转换为模型输入格式: (1, Channels, Time) -> 注意这里还需要 transpose
    # make_data 里的 Dataset 做了一次 transpose(0,1)，所以这里也要对齐
    # 特征 shape: (300, 40)
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32)  # (Time, Feat)
    input_tensor = input_tensor.transpose(0, 1)  # (Feat, Time) -> (40, 300)
    input_tensor = input_tensor.unsqueeze(0)  # (Batch, Feat, Time) -> (1, 40, 300)

    # 模型推理
    model = load_model()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_label = np.argmax(probs)

    # 4. 展示结果
    with col2:
        st.subheader("2. 识别结果")

        # 结果大字展示
        emotion_name = EMOTION_LABELS.get(pred_label, "未知")
        confidence = probs[pred_label] * 100

        if confidence > 60:
            st.success(f"识别情感：**{emotion_name}** (置信度: {confidence:.1f}%)")
        else:
            st.warning(f"识别情感：**{emotion_name}** (置信度较低: {confidence:.1f}%)")

        # 概率分布柱状图
        st.write("各情感概率分布：")
        chart_data = {label: prob for label, prob in zip(EMOTION_LABELS.values(), probs)}
        st.bar_chart(chart_data)

    # 5. 特征可视化 (声谱图)
    st.write("---")
    st.subheader("3. 深度特征可视化 (MFCC 热力图)")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mfcc_features.T, x_axis='time', ax=ax_spec, cmap='viridis')
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB")
    ax_spec.set_title("MFCC Spectrogram")
    st.pyplot(fig_spec)
