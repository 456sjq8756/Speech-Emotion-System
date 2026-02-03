import streamlit as st
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from model import CNN_LSTM_Model
import os

# --- 1. 配置参数 ---
INPUT_SIZE = 40
HIDDEN_SIZE = 128
NUM_CLASSES = 6
DURATION = 3
SAMPLE_RATE = 22050
DEVICE = torch.device('cpu')

# 情感标签
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
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            return model
        except Exception as e:
            st.error(f"模型加载出错: {e}")
            return None
    else:
        st.error("⚠️ 找不到 'best_model.pth'。请先运行 train.py 进行训练！")
        return None


# --- 3. 预处理函数 ---
def preprocess_audio(y, sr):
    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=INPUT_SIZE)
    return mfcc


# --- 4. 雷达图绘制 ---
def plot_radar_chart(probs):
    categories = list(EMOTION_LABELS.values())
    values = list(probs)
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='情感概率',
        line_color='#FF4B4B',
        fillcolor='rgba(255, 75, 75, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title={'text': "📊 情感概率分布", 'y': 0.95, 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# --- 5. 核心分析逻辑 ---
def analyze_audio(audio_source):
    # 加载音频 (兼容文件上传 和 录音的 BytesIO)
    try:
        y, sr = librosa.load(audio_source, sr=SAMPLE_RATE, duration=DURATION)
    except Exception as e:
        st.error(f"音频解析失败: {e}")
        return

    # 隐私模式
    if privacy_mode:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
        st.toast("已应用隐私脱敏处理", icon="🛡️")

    # 布局
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. 音频分析")
        st.audio(y, sample_rate=sr)

        fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#1f77b4')
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)

        st.markdown("**MFCC 特征**")
        mfcc_features = preprocess_audio(y, sr)
        fig_spec, ax_spec = plt.subplots(figsize=(6, 2))
        img = librosa.display.specshow(mfcc_features, x_axis='time', ax=ax_spec, cmap='inferno')
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB")
        st.pyplot(fig_spec)

    with col2:
        st.subheader("2. 识别结果")
        input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)

        model = load_model()
        if model:
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)

            top_emotion = EMOTION_LABELS[pred_idx]
            confidence = probs[pred_idx] * 100

            if confidence > 70:
                st.success(f"### 🎯 识别情感：{top_emotion}")
            elif confidence > 40:
                st.warning(f"### ⚠️ 识别情感：{top_emotion}")
            else:
                st.error(f"### ❓ 识别情感：{top_emotion}")

            st.write(f"**置信度:** {confidence:.2f}%")
            st.plotly_chart(plot_radar_chart(probs), use_container_width=True)


# --- 6. 页面主入口 ---
st.set_page_config(page_title="语音情感识别系统", layout="wide", page_icon="🎙️")

st.title("🎙️ 基于 CNN-LSTM-Attention 的语音情感识别系统")
st.markdown("### 🎓 本科毕业设计演示 | 邵金桥")
st.write("---")

with st.sidebar:
    st.header("⚙️ 系统设置")
    privacy_mode = st.checkbox("🛡️ 开启隐私保护模式", value=False)
    st.info("💡 **说明**：\n可选择上传文件或直接使用麦克风录音。")

tab1, tab2 = st.tabs(["📂 上传文件模式", "🎤 实时录音模式"])

# --- Tab 1: 上传文件 ---
with tab1:
    uploaded_file = st.file_uploader("请上传一段语音文件 (.wav)", type=['wav'])
    if uploaded_file is not None:
        analyze_audio(uploaded_file)

# --- Tab 2: 实时录音 (使用官方原生组件) ---
with tab2:
    st.write("点击下方红色按钮开始录音：")
    # ✨✨✨ 重点：直接使用 st.audio_input，不需要安装任何库！ ✨✨✨
    audio_value = st.audio_input("按住录音")

    if audio_value:
        st.success("✅ 录音完成，正在分析...")
        analyze_audio(audio_value)
