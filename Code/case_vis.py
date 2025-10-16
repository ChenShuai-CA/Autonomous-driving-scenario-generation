import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np  # 添加 numpy 以确保兼容性

# 加载数据
file_path = "/media/shuai/DATA2/-Accident-prone-traffic-scenario-generation/Code/data/Interation/DR_USA_Intersection_EP1/train/DR_USA_Intersection_EP1_train_case_4.txt"

# 调试文件路径
print(f"Loading data from: {file_path}")

# 尝试读取数据并捕获潜在错误
try:
    data = pd.read_csv(file_path, sep="\t", header=None, names=["frame_ID", "agent_ID", "pos_x", "pos_y", "agent_type"])
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# 绘制静态图
def plot_static_frame(frame_id):
    frame_data = data[data["frame_ID"] == frame_id]
    plt.figure(figsize=(10, 6))
    for agent_type, group in frame_data.groupby("agent_type"):
        plt.scatter(group["pos_x"], group["pos_y"], label=agent_type, s=50)
    plt.title(f"Frame {frame_id}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

# 生成动画
def plot_animation():
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter([], [], s=50)

    def init():
        ax.set_xlim(data["pos_x"].min() - 10, data["pos_x"].max() + 10)
        ax.set_ylim(data["pos_y"].min() - 10, data["pos_y"].max() + 10)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Dynamic Trajectory Animation")
        return scatter,

    def update(frame_id):
        frame_data = data[data["frame_ID"] == frame_id]
        scatter.set_offsets(frame_data[["pos_x", "pos_y"]].values)
        scatter.set_array(frame_data["agent_type"].astype("category").cat.codes)
        return scatter,

    ani = FuncAnimation(fig, update, frames=sorted(data["frame_ID"].unique()), init_func=init, blit=True)
    plt.show()

# 示例：绘制第1帧的静态图
plot_static_frame(1)

# 示例：生成动画
plot_animation()