import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class APF_Robot:
    def __init__(self, start, goal, obstacles, k_att=4.0, k_rep=100.0, rr=2.0, step_size=0.1, enable_escape=False):
        # k_att 调大到 4.0，让它更渴望到达目标
        # rr 调小到 2.0，让斥力场更紧凑，留出路给它走
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr
        self.step_size = step_size
        
        self.enable_escape = enable_escape
        self.path = [self.pos.copy()]
        self.is_reached = False
        
        # 状态机
        self.stuck_counter = 0
        self.escape_timer = 0
        self.escape_direction = np.zeros(2)

    def calculate_repulsive_force(self):
        f_rep = np.zeros(2)
        for (ox, oy, r) in self.obstacles:
            obs_pos = np.array([ox, oy])
            dist_vec = self.pos - obs_pos
            dist = np.linalg.norm(dist_vec)
            
            if dist <= self.rr:
                rep_val = self.k_rep * (1.0/dist - 1.0/self.rr) * (1.0/(dist**2))
                f_rep += rep_val * (dist_vec / dist)
        return f_rep

    def calculate_attractive_force(self):
        return -self.k_att * (self.pos - self.goal)

    def step(self):
        if self.is_reached: return

        f_rep = self.calculate_repulsive_force()

        # --- 模式 A: 逃逸模式 (侧向滑步) ---
        if self.escape_timer > 0:
            self.escape_timer -= 1
            
            # 策略：只保留“侧向逃逸力” + “斥力(防止撞墙)”
            # 暂时切断引力，防止被吸回坑里
            f_total = self.escape_direction + f_rep
            
            if self.escape_timer == 0:
                print("✅ 逃离完成，重新寻找目标")

        # --- 模式 B: 正常导航 ---
        else:
            f_att = self.calculate_attractive_force()
            f_total = f_att + f_rep
            
            # 死锁检测
            if self.enable_escape:
                f_norm = np.linalg.norm(f_total)
                # 检测逻辑：如果停滞不前 且 还没到终点
                if f_norm < 10 and np.linalg.norm(self.pos - self.goal) > 1.0:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                
                if self.stuck_counter > 15:
                    print("⚠️ 此路不通！执行切向战术机动...")
                    self.escape_timer = 45 # 持续 45 帧
                    self.stuck_counter = 0
                    
                    # --- 智能逃逸方向计算 ---
                    # 既然被卡住，说明阻力主要来自前方 (X轴方向)
                    # 我们就强制往侧方 (Y轴) 移动
                    # 如果当前在 Y>0，就往上跑；如果在 Y<0，就往下跑
                    if self.pos[1] >= 0:
                        self.escape_direction = np.array([0.2, 1.0]) * 60 # 向上偏右一点
                    else:
                        self.escape_direction = np.array([0.2, -1.0]) * 60 # 向下偏右一点
                    
                    f_total = self.escape_direction + f_rep

        # 物理移动
        f_norm_final = np.linalg.norm(f_total)
        if f_norm_final > 0:
            # 限制最大单步速度，防止瞬移
            step = self.step_size if self.escape_timer == 0 else self.step_size * 1.2
            self.pos += (f_total / f_norm_final) * step
        
        self.pos[0] = np.clip(self.pos[0], -2, 14)
        self.pos[1] = np.clip(self.pos[1], -6, 6)

        self.path.append(self.pos.copy())
        if np.linalg.norm(self.pos - self.goal) < 0.3:
            self.is_reached = True
            print("🎉 目标到达！")

class APF_Demo_GUI:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.2)
        self.ax.set_title("APF Path Planning Demo")
        
        self.robot = None
        self.anim = None
        
        self.robot_point, = self.ax.plot([], [], 'go', markersize=10, zorder=5, label='Robot')
        self.traj_line, = self.ax.plot([], [], 'g-', linewidth=1, zorder=4, label='Path')
        self.goal_point, = self.ax.plot([], [], 'r*', markersize=15, zorder=5, label='Goal')
        self.obstacles_patches = []
        self.range_patches = []

        ax_btn1 = plt.axes([0.1, 0.05, 0.2, 0.075])
        ax_btn2 = plt.axes([0.4, 0.05, 0.2, 0.075])
        ax_btn3 = plt.axes([0.7, 0.05, 0.2, 0.075])

        self.btn1 = Button(ax_btn1, '1. Basic', color='lightblue', hovercolor='0.95')
        self.btn2 = Button(ax_btn2, '2. Trap (Fail)', color='salmon', hovercolor='0.95')
        self.btn3 = Button(ax_btn3, '3. Smart Escape', color='lightgreen', hovercolor='0.95')

        self.btn1.on_clicked(self.load_basic)
        self.btn2.on_clicked(self.load_trap)
        self.btn3.on_clicked(self.load_escape)

        self.load_basic(None)

    def reset_plot(self):
        self.ax.set_xlim(-2, 14)
        self.ax.set_ylim(-5, 5) # 稍微缩小视野，让物体看起来更大更清楚
        self.ax.grid(True)
        for p in self.obstacles_patches: p.remove()
        for p in self.range_patches: p.remove()
        self.obstacles_patches = []
        self.range_patches = []

    def draw_static(self):
        self.goal_point.set_data([self.robot.goal[0]], [self.robot.goal[1]])
        for (ox, oy, r) in self.robot.obstacles:
            c = plt.Circle((ox, oy), r/2, color='#444444', alpha=0.8)
            self.ax.add_patch(c)
            self.obstacles_patches.append(c)
            # 绘制斥力范围
            c_range = plt.Circle((ox, oy), self.robot.rr, color='r', fill=False, linestyle='--', alpha=0.2)
            self.ax.add_patch(c_range)
            self.range_patches.append(c_range)

    def restart_anim(self):
        if self.anim: self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self.update, frames=800, interval=15, blit=True)
        plt.draw()

    def update(self, frame):
        if self.robot and not self.robot.is_reached:
            self.robot.step()
            path = np.array(self.robot.path)
            self.robot_point.set_data([self.robot.pos[0]], [self.robot.pos[1]])
            self.traj_line.set_data(path[:, 0], path[:, 1])
        return self.robot_point, self.traj_line, self.goal_point

    def load_basic(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 1: Basic Obstacles")
        obs = [[4, 0.5, 2.0], [8, -1.5, 2.0], [6, 3, 1.5]]
        # rr=2.5 适中
        self.robot = APF_Robot([0, 0], [12, 0], obs, rr=2.5, enable_escape=False)
        self.draw_static()
        self.restart_anim()

    def load_trap(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 2: Local Minima (Stuck)")
        # 设计一个更紧凑的陷阱，但把斥力圈 rr 缩小到 2.2，
        # 这样中间虽过不去，但两边是有“缝隙”可以绕的
        obs = [
            [6.0, 1.8, 1.5],  # 上
            [6.0, -1.8, 1.5], # 下
            [7.5, 0, 1.8]     # 中后
        ]
        self.robot = APF_Robot([0, 0], [12, 0], obs, k_rep=120, rr=2.2, enable_escape=False)
        self.draw_static()
        self.restart_anim()

    def load_escape(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 3: Smart Escape Strategy")
        # 一模一样的陷阱
        obs = [
            [6.0, 1.8, 1.5],
            [6.0, -1.8, 1.5],
            [7.5, 0, 1.8]
        ]
        # 开启 enable_escape
        self.robot = APF_Robot([0, 0], [12, 0], obs, k_rep=120, rr=2.2, enable_escape=True)
        self.draw_static()
        self.restart_anim()

if __name__ == "__main__":
    gui = APF_Demo_GUI()
    plt.show()