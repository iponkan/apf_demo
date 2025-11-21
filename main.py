import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class APF_Robot:
    def __init__(self, start, goal, obstacles, k_att=5.0, k_rep=100.0, rr=2.0, step_size=0.1):
        """
        :param start: 起点 [x, y]
        :param goal: 终点 [x, y]
        :param obstacles: 障碍物列表 [[x, y, radius], ...]
        :param k_att: 引力增益系数
        :param k_rep: 斥力增益系数
        :param rr: 斥力影响半径 (Repulsive Radius)
        :param step_size: 模拟步长
        """
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr
        self.step_size = step_size
        
        self.path = [self.pos.copy()]
        self.is_reached = False

    def calculate_attractive_force(self):
        """计算引力: F_att = -k_att * (q - q_goal)"""
        return -self.k_att * (self.pos - self.goal)

    def calculate_repulsive_force(self):
        """计算斥力: F_rep"""
        f_rep = np.zeros(2)
        for (ox, oy, r) in self.obstacles:
            obs_pos = np.array([ox, oy])
            # 机器人到障碍物中心的距离
            dist_vec = self.pos - obs_pos
            dist = np.linalg.norm(dist_vec)
            
            # 实际距离要减去障碍物自身的物理半径（视为圆形障碍物）
            # 为了简化教学，这里假设障碍物是点，或者dist已经是表面距离
            # 这里我们把dist视为到障碍物中心的距离，如果小于影响范围则产生斥力
            
            if dist <= self.rr:
                # 斥力方向：指向机器人（远离障碍物）
                # 标准 APF 斥力公式求导后的力向量
                rep_val = self.k_rep * (1.0/dist - 1.0/self.rr) * (1.0/(dist**2))
                f_rep += rep_val * (dist_vec / dist)
                
        return f_rep

    def step(self):
        """执行一步移动"""
        if self.is_reached:
            return

        # 1. 计算合力
        f_att = self.calculate_attractive_force()
        f_rep = self.calculate_repulsive_force()
        f_total = f_att + f_rep

        # 2. 归一化合力方向并移动 (保持速度恒定，便于观察轨迹)
        # 在实际物理中，力决定加速度，这里简化为决定速度方向
        f_norm = np.linalg.norm(f_total)
        if f_norm > 0:
            direction = f_total / f_norm
            self.pos += direction * self.step_size
        
        self.path.append(self.pos.copy())

        # 3. 判断是否到达目标 (距离小于阈值)
        if np.linalg.norm(self.pos - self.goal) < 0.2:
            self.is_reached = True
            print("🎉 目标已到达！")

# --- 可视化设置 ---
def run_simulation():
    # 1. 设置场景
    start_pos = [0, 0]
    goal_pos = [10, 10]
    # 障碍物: [x, y, 绘图半径]
    obstacles = [
        [3, 2, 1],
        [6, 5, 1.5],
        [8, 9, 1],
        [4, 7, 1]
    ]

    robot = APF_Robot(start_pos, goal_pos, obstacles, k_att=1.0, k_rep=20.0, rr=3.0, step_size=0.1)

    # 2. 初始化绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_title("Artificial Potential Field (APF) Simulation")
    ax.grid(True)

    # 绘制固定元素
    ax.plot(start_pos[0], start_pos[1], 'bs', label='Start', markersize=10)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', label='Goal', markersize=15)
    
    # 绘制障碍物 (圆圈)
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r/2, color='k', fill=True, alpha=0.5) # r/2 只是为了绘图好看
        ax.add_patch(circle)
        # 画出斥力影响范围 (虚线圆)
        limit_circle = plt.Circle((ox, oy), 3.0, color='r', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(limit_circle)

    # 绘制机器人和轨迹
    robot_point, = ax.plot([], [], 'go', markersize=8, label='Robot')
    trajectory, = ax.plot([], [], 'g-', linewidth=1, label='Path')
    
    ax.legend(loc='upper left')

    # 3. 动画更新函数
    def update(frame):
        if not robot.is_reached:
            robot.step()
        
        # 获取当前路径数据
        path_arr = np.array(robot.path)
        robot_point.set_data([robot.pos[0]], [robot.pos[1]]) # 必须是序列
        trajectory.set_data(path_arr[:, 0], path_arr[:, 1])
        return robot_point, trajectory

    # 4. 启动动画
    # interval=30 表示每30ms刷新一帧
    anim = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    print("开始仿真... 关闭窗口以退出。")
    plt.show()

if __name__ == "__main__":
    run_simulation()