import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class APF_Robot:
    def __init__(self, start, goal, obstacles, k_att=2.0, k_rep=80.0, rr=3.0, step_size=0.1, enable_escape=False):
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
        
        # 状态机变量
        self.stuck_counter = 0
        self.escape_timer = 0
        self.escape_force = np.zeros(2)

    def calculate_repulsive_force(self):
        """单独计算斥力"""
        f_rep = np.zeros(2)
        for (ox, oy, r) in self.obstacles:
            obs_pos = np.array([ox, oy])
            dist_vec = self.pos - obs_pos
            dist = np.linalg.norm(dist_vec)
            
            if dist <= self.rr:
                # 斥力公式
                rep_val = self.k_rep * (1.0/dist - 1.0/self.rr) * (1.0/(dist**2))
                f_rep += rep_val * (dist_vec / dist)
        return f_rep

    def calculate_attractive_force(self):
        """单独计算引力"""
        return -self.k_att * (self.pos - self.goal)

    def step(self):
        if self.is_reached: return

        f_rep = self.calculate_repulsive_force()

        # --- 模式 A: 逃逸模式 (Escape Mode) ---
        if self.escape_timer > 0:
            # 关键改进：逃逸时，保留斥力！
            # 这样机器人如果随机撞向墙壁，斥力会把它推开，从而实现“沿墙滑行”的效果
            f_total = self.escape_force + f_rep 
            
            self.escape_timer -= 1
            if self.escape_timer == 0:
                print("✅ 逃逸结束，恢复正常导航")

        # --- 模式 B: 正常导航模式 ---
        else:
            f_att = self.calculate_attractive_force()
            f_total = f_att + f_rep
            f_norm = np.linalg.norm(f_total)

            # 死锁检测
            if self.enable_escape:
                # 如果合力很小（陷入力平衡）或者 距离目标很远却停滞不前
                if f_norm < 10 and np.linalg.norm(self.pos - self.goal) > 1.0:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                
                # 触发逃逸 (增加敏感度，连续 10 帧卡住就触发)
                if self.stuck_counter > 10:
                    print("⚠️ 检测到死锁，启动强力逃逸！")
                    self.escape_timer = 60 # 增加逃逸时间到 60 帧 (约 1.2秒)
                    self.stuck_counter = 0
                    
                    # 生成随机方向，但力度一定要大
                    rand_angle = np.random.uniform(0, 2*np.pi)
                    direction = np.array([np.cos(rand_angle), np.sin(rand_angle)])
                    
                    # 技巧：给一个巨大的力，确保暂时忽略引力的影响
                    self.escape_force = direction * 150 

                    # 立即应用新力
                    f_total = self.escape_force + f_rep

        # 物理移动更新
        f_norm_final = np.linalg.norm(f_total)
        if f_norm_final > 0:
            # 限制最大步长，保证动画平滑
            step = self.step_size
            # 如果是在逃逸，稍微跑快一点
            if self.escape_timer > 0:
                step = self.step_size * 1.5
            
            self.pos += (f_total / f_norm_final) * step
        
        # 边界限制
        self.pos[0] = np.clip(self.pos[0], -2, 14)
        self.pos[1] = np.clip(self.pos[1], -6, 6)

        self.path.append(self.pos.copy())
        
        if np.linalg.norm(self.pos - self.goal) < 0.2:
            self.is_reached = True
            print("🎉 目标到达！")


# --- GUI 管理器 ---
class APF_Demo_GUI:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.2)
        self.ax.set_title("Artificial Potential Field (APF) Teaching Demo")
        
        self.robot = None
        self.anim = None
        
        # 绘图元素
        self.robot_point, = self.ax.plot([], [], 'go', markersize=10, zorder=5, label='Robot')
        self.traj_line, = self.ax.plot([], [], 'g-', linewidth=1, zorder=4, label='Path')
        self.goal_point, = self.ax.plot([], [], 'r*', markersize=15, zorder=5, label='Goal')
        self.obstacles_patches = []
        self.range_patches = []

        # 按钮
        ax_btn1 = plt.axes([0.1, 0.05, 0.2, 0.075])
        ax_btn2 = plt.axes([0.4, 0.05, 0.2, 0.075])
        ax_btn3 = plt.axes([0.7, 0.05, 0.2, 0.075])

        self.btn1 = Button(ax_btn1, 'Scenario 1:\nBasic', color='lightblue', hovercolor='0.975')
        self.btn2 = Button(ax_btn2, 'Scenario 2:\nTrap (Fail)', color='salmon', hovercolor='0.975')
        self.btn3 = Button(ax_btn3, 'Scenario 3:\nEscape (Success)', color='lightgreen', hovercolor='0.975')

        self.btn1.on_clicked(self.load_scenario_basic)
        self.btn2.on_clicked(self.load_scenario_trap)
        self.btn3.on_clicked(self.load_scenario_escape)

        self.load_scenario_basic(None)

    def reset_plot(self):
        self.ax.set_xlim(-2, 14)
        self.ax.set_ylim(-6, 6)
        self.ax.grid(True)
        for p in self.obstacles_patches: p.remove()
        for p in self.range_patches: p.remove()
        self.obstacles_patches = []
        self.range_patches = []

    def draw_static_elements(self):
        self.goal_point.set_data([self.robot.goal[0]], [self.robot.goal[1]])
        for (ox, oy, r) in self.robot.obstacles:
            c = plt.Circle((ox, oy), r/2, color='#555555', alpha=0.9) # 深灰色障碍物
            self.ax.add_patch(c)
            self.obstacles_patches.append(c)
            c_range = plt.Circle((ox, oy), self.robot.rr, color='r', fill=False, linestyle='--', alpha=0.2)
            self.ax.add_patch(c_range)
            self.range_patches.append(c_range)

    def restart_animation(self):
        if self.anim is not None: self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self.update, frames=600, interval=20, blit=True) # 增加总帧数
        plt.draw()

    def update(self, frame):
        if self.robot and not self.robot.is_reached:
            self.robot.step()
            path = np.array(self.robot.path)
            self.robot_point.set_data([self.robot.pos[0]], [self.robot.pos[1]])
            self.traj_line.set_data(path[:, 0], path[:, 1])
        return self.robot_point, self.traj_line, self.goal_point

    # --- 场景定义 ---
    def load_scenario_basic(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 1: Basic Obstacle Avoidance")
        start = [0, 0]
        goal = [12, 0]
        # 简单的散乱障碍物
        obs = [[4, 0.5, 2], [8, -1, 2], [6, 3, 1.5]]
        self.robot = APF_Robot(start, goal, obs, enable_escape=False)
        self.draw_static_elements()
        self.restart_animation()

    def load_scenario_trap(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 2: Local Minima Trap (Robot gets stuck)")
        start = [0, 0]
        goal = [12, 0]
        
        # --- 改进的陷阱设计 ---
        # 把 U 型口稍微张开一点，不要封死
        obs = [
            [6.5, 2.5, 1.5], # 上方
            [6.5, -2.5, 1.5], # 下方
            [8.0, 0, 2.0]    # 正后方大石头
        ]
        # 即使只有这三个，由于斥力场半径很大(rr=3.5)，中间依然是过不去的
        self.robot = APF_Robot(start, goal, obs, k_rep=80.0, rr=4.0, enable_escape=False)
        self.draw_static_elements()
        self.restart_animation()

    def load_scenario_escape(self, event):
        self.reset_plot()
        self.ax.set_title("Scenario 3: Improved APF (Escape Strategy)")
        start = [0, 0]
        goal = [12, 0]
        
        # 使用完全相同的陷阱
        obs = [
            [6.5, 2.5, 1.5],
            [6.5, -2.5, 1.5],
            [8.0, 0, 2.0]
        ]
        
        # 开启逃逸
        self.robot = APF_Robot(start, goal, obs, k_rep=80.0, rr=4.0, enable_escape=True)
        self.draw_static_elements()
        self.restart_animation()

if __name__ == "__main__":
    gui = APF_Demo_GUI()
    plt.show()