import streamlit as st
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from model import CNN_LSTM_Model, CNN_Model, LSTM_Model
from dataset import load_data

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="语音情感分析系统", layout="wide", page_icon="🎙️")

# --- 2. 自定义 CSS 样式 ---
st.markdown("""
<style>
    .stApp { background-color: #F4F7FB; }
    .main-title {
        font-size: 38px; font-weight: 900;
        background: -webkit-linear-gradient(45deg, #4A90E2, #9013FE);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 5px;
    }
    .sub-title { text-align: center; color: #666666; font-size: 16px; margin-bottom: 30px; }

    .result-card {
        background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
        border-radius: 15px; padding: 25px 20px; color: white;
        text-align: center; box-shadow: 0 8px 16px rgba(110, 142, 251, 0.3);
        margin-bottom: 20px;
    }
    .result-card h4 { margin: 0; font-size: 16px; font-weight: 500; opacity: 0.9; }
    .result-card h1 { margin: 10px 0; font-size: 42px; font-weight: bold; color: white; }
    .result-card p { margin: 0; font-size: 15px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- 3. 参数与映射 ---
INPUT_SIZE = 40
HIDDEN_SIZE = 128
NUM_CLASSES = 6
DURATION = 3
SAMPLE_RATE = 22050
DEVICE = torch.device('cpu')

EMOTION_LABELS = {
    0: '😡 愤怒 (Angry)', 1: '😨 恐惧 (Fear)', 2: '😂 快乐 (Happy)',
    3: '😐 中性 (Neutral)', 4: '😭 悲伤 (Sad)', 5: '😲 惊讶 (Surprise)'
}


# --- 4. 模型加载与数据处理函数 ---
@st.cache_resource
def load_main_model():
    model = CNN_LSTM_Model(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        model.eval()
        return model
    return None


def preprocess_audio(y, sr):
    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=INPUT_SIZE)


def plot_radar_chart(probs):
    categories = list(EMOTION_LABELS.values())
    values = list(probs)

    # 闭合雷达图首尾
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line_color='#a777e3', fillcolor='rgba(167, 119, 227, 0.4)',
        marker=dict(size=8, color='#6e8efb')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#EAEAEA', tickfont=dict(color='#888')),
            angularaxis=dict(tickfont=dict(size=14, color='#333', family="Arial, sans-serif"))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=30, b=30), height=450
    )
    return fig


# --- 5. 页面顶部与侧边栏 ---
st.markdown('<p class="main-title">基于 CNN-LSTM 的语音情感识别系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">🎓 本科毕业设计演示 | 邵金桥</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
    st.header("🎯 系统导航")
    page = st.radio("请选择功能模块:", ["🎧 语音分析工作台", "📊 模型基线评估报告"])

    st.markdown("---")

    st.header("⚙️ 工作原理")
    st.markdown("""
    - 🎵 **提取 MFCC 频谱特征**
    - 🧠 **CNN 提取局部特征**
    - ⏱️ **LSTM 捕捉时序动态**
    - 🎯 **Attention 聚焦情感关键帧**
    """)

    st.markdown("---")
    privacy_mode = st.toggle("🛡️ 开启隐私脱敏", value=False, help="开启后将对输入语音进行音高变换，保护声纹隐私。")

# --- 6. 核心功能页：语音分析工作台 ---
if page == "🎧 语音分析工作台":

    st.markdown("### 📂 音频输入区")
    tab1, tab2 = st.tabs(["📄 本地文件上传", "🎤 麦克风实时录音"])


    def analyze_audio_ui(audio_source, file_details=None):
        y, sr = librosa.load(audio_source, sr=SAMPLE_RATE, duration=DURATION)

        # VAD 处理
        y_trimmed, index = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) > sr * 0.5:
            y = y_trimmed
        elif len(y_trimmed) <= sr * 0.5:
            st.warning("⚠️ 语音过短或声音过小，可能影响精度。")

        if privacy_mode:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)

        if file_details:
            st.info(f"**当前文件:** {file_details['name']} | **大小:** {file_details['size']}")
        st.audio(y, sample_rate=sr)
        st.markdown("---")

        # 特征提取与模型推理
        mfcc_features = preprocess_audio(y, sr)
        input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        model = load_main_model()

        probs = np.zeros(NUM_CLASSES)
        pred_idx = 0
        if model:
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)

        top_emotion = EMOTION_LABELS[pred_idx]
        confidence = probs[pred_idx] * 100

        # 左右完美对齐的布局
        col1, col2 = st.columns([1.1, 1], gap="large")

        with col1:
            st.markdown("### 🎵 声学特征与波形")

            # 第一张图：波形图
            fig_wave, ax_wave = plt.subplots(figsize=(8, 2.5))
            fig_wave.patch.set_facecolor('#FFFFFF')
            ax_wave.set_facecolor('#F8F9FA')
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#6e8efb', alpha=0.9)
            ax_wave.set_title("VAD 时域波形图", fontsize=11, color='#555')
            ax_wave.spines['top'].set_visible(False)
            ax_wave.spines['right'].set_visible(False)
            st.pyplot(fig_wave)

            # 核心对齐间距
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

            # 第二张图：频谱图
            fig_spec, ax_spec = plt.subplots(figsize=(8, 2.5))
            fig_spec.patch.set_facecolor('#FFFFFF')
            img = librosa.display.specshow(mfcc_features, x_axis='time', ax=ax_spec, cmap='magma')
            ax_spec.set_title("MFCC 频谱特征分布", fontsize=11, color='#555')
            fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
            st.pyplot(fig_spec)

        with col2:
            st.markdown("### 🤖 模型识别结果")

            # 渐变卡片
            st.markdown(f"""
            <div class="result-card">
                <h4>系统首选预测情感</h4>
                <h1>{top_emotion}</h1>
                <p>综合置信度: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🎯 置信度分数详情")

            # 纯 HTML/CSS 渲染弹性进度条
            progress_html = ""
            for i in range(NUM_CLASSES):
                emo_name = EMOTION_LABELS[i]
                score = probs[i] * 100
                bar_color = "#4CAF50" if i == pred_idx else "#D3D3D3"
                font_weight = "bold" if i == pred_idx else "normal"
                text_color = "#333333" if i == pred_idx else "#777777"

                progress_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 14px;">
                    <div style="width: 32%; font-size: 14px; font-weight: {font_weight}; color: {text_color};">
                        {emo_name}
                    </div>
                    <div style="width: 53%; background-color: #F0F0F0; border-radius: 6px; height: 10px; margin: 0 10px;">
                        <div style="width: {score}%; height: 10px; background-color: {bar_color}; border-radius: 6px;"></div>
                    </div>
                    <div style="width: 15%; text-align: right; font-size: 14px; font-weight: {font_weight}; color: {text_color};">
                        {score:.1f}%
                    </div>
                </div>
                """
            st.markdown(progress_html, unsafe_allow_html=True)

        # 横跨底部的雷达图
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #333;'>🕸️ 情感多维雷达分布</h3>", unsafe_allow_html=True)
        st.plotly_chart(plot_radar_chart(probs), use_container_width=True)


    with tab1:
        uploaded_file = st.file_uploader("", type=['wav'], help="支持上传 .wav 格式的语音文件")
        if uploaded_file is not None:
            details = {"name": uploaded_file.name, "size": f"{uploaded_file.size / 1024:.1f} KB"}
            analyze_audio_ui(uploaded_file, details)

    with tab2:
        audio_value = st.audio_input("点击按钮进行录音测试")
        if audio_value:
            details = {"name": "实时麦克风语音", "size": "N/A"}
            analyze_audio_ui(audio_value, details)

# --- 7. 评估报告页：包含消融实验与训练日志 ---
elif page == "📊 模型基线评估报告":
    st.subheader("📚 课题模型基线评估与实验分析")
    st.write(
        "本系统在相同数据集上分别训练了纯 CNN、纯 LSTM 以及本课题提出的 **CNN-LSTM 混合模型**，以下为测试集上的评估结果。")


    @st.cache_data(show_spinner=False)
    def evaluate_all_models():
        _, test_dataset = load_data()
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        models_dict = {
            "纯 CNN 模型": (CNN_Model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES), 'cnn_model.pth'),
            "纯 LSTM 模型": (LSTM_Model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES), 'lstm_model.pth'),
            "CNN-LSTM (本项目)": (CNN_LSTM_Model(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES), 'best_model.pth')
        }
        results_acc = {}
        best_cm = None
        for name, (model, path) in models_dict.items():
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                correct, total = 0, 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for feats, labels in test_loader:
                        outputs = model(feats)
                        _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        if name == "CNN-LSTM (本项目)":
                            all_preds.extend(preds.numpy())
                            all_labels.extend(labels.numpy())
                results_acc[name] = (correct / total) * 100
                if name == "CNN-LSTM (本项目)":
                    best_cm = confusion_matrix(all_labels, all_preds)
            else:
                results_acc[name] = 0.0
        return results_acc, best_cm


    if st.button("🚀 加载测试集并生成实验报告", type="primary"):
        with st.spinner("🧠 正在加载测试集并进行模型推理，请稍候..."):
            accuracies, best_cm = evaluate_all_models()
        st.success("✅ 测试集评估完成！")

        col1, col2, col3 = st.columns(3)
        col1.metric("纯 CNN 模型准确率", f"{accuracies.get('纯 CNN 模型', 0):.2f}%")
        col2.metric("纯 LSTM 模型准确率", f"{accuracies.get('纯 LSTM 模型', 0):.2f}%")
        col3.metric("🏆 CNN-LSTM (本项目)", f"{accuracies.get('CNN-LSTM (本项目)', 0):.2f}%", "表现最优")
        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            df_acc = pd.DataFrame(list(accuracies.items()), columns=['模型', '准确率 (%)'])
            fig_bar = px.bar(df_acc, x='模型', y='准确率 (%)', color='模型',
                             color_discrete_sequence=['#B0BEC5', '#B0BEC5', '#6e8efb'], text='准确率 (%)',
                             title="多模型性能对比柱状图")
            fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            if best_cm is not None:
                labels_list = [val.split(' ')[0] for val in EMOTION_LABELS.values()]
                fig_cm = px.imshow(best_cm, text_auto=True, x=labels_list, y=labels_list,
                                   color_continuous_scale='Blues', aspect="auto", title="CNN-LSTM 模型混淆矩阵")

                # ✨ 核心修改：明确加上 X 轴和 Y 轴的含义 (预测标签 / 真实标签)
                fig_cm.update_layout(
                    xaxis_title="<b>预测标签 (Predicted)</b>",
                    yaxis_title="<b>真实标签 (True)</b>"
                )

                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        log_file = 'best_model_log.csv'
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            curve_col1, curve_col2 = st.columns(2)
            with curve_col1:
                fig_loss = px.line(df_log, x='Epoch', y='Loss', title='训练误差下降曲线 (Training Loss)', markers=True)
                fig_loss.update_traces(line_color='#FF6584')
                st.plotly_chart(fig_loss, use_container_width=True)
            with curve_col2:
                fig_acc = px.line(df_log, x='Epoch', y='Accuracy', title='验证集准确率上升曲线 (Validation Accuracy)',
                                  markers=True)
                fig_acc.update_traces(line_color='#6e8efb')
                st.plotly_chart(fig_acc, use_container_width=True)
