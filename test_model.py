import torch
from model import CNN_LSTM_Model

# 1. 模拟一个输入数据
# 假设 batch_size=4, 特征数=40 (MFCC), 时间步长=300 (约3秒音频)
fake_input = torch.randn(4, 40, 300)

# 2. 实例化模型
# 假设我们要分类 6 种情感 (CASIA数据集通常是6类)
# 显式传入 hidden_size=128
model = CNN_LSTM_Model(input_size=40, hidden_size=128, num_classes=6)

print("正在测试模型前向传播...")

# 3. 运行模型
try:
    output = model(fake_input)
    print("\n✅ 模型测试成功！")
    print("输入尺寸:", fake_input.shape)
    print("输出尺寸:", output.shape)
    print("期望输出: [4, 6] (Batch Size, 类别数)")

    if output.shape == (4, 6):
        print("\n🎉 结构验证通过！可以直接用于毕设。")
    else:
        print("\n⚠️ 尺寸不对，请检查代码。")

except Exception as e:
    print("\n❌ 报错了:", e)
