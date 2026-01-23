#!/usr/bin/env python3
"""
机器人在图书馆四处游荡的控制脚本

功能：
    - 控制机器人在图书馆内随机移动
    - 遇到障碍物时向左旋转120度继续运动
    - 实时可视化机器人运动状态
    - 提供详细的运行日志和进度信息

依赖库：
    - mujoco：MuJoCo物理引擎的官方Python绑定
    - numpy：用于数学计算和向量操作
    - time：用于时间控制和进度计算

使用方法：
    1. 确保已安装MuJoCo 3.4.0及以上版本
    2. 确保已安装所需依赖库：pip install mujoco numpy
    3. 确保bookstore-lounge.xml模型文件存在于当前目录
    4. 运行脚本：python robot_wanderer.py

配置参数：
    - speed：移动速度（默认0.5，可在初始化时调整）
    - turn_angle：转向角度（弧度，默认120度=2π/3，可自定义）
    - collision_threshold：碰撞检测阈值（默认0.1，可在初始化时调整）
    - move_duration：每次移动持续时间（默认2秒，可在初始化时调整）
    - wander_duration：总游荡时间（默认60秒，可在调用wander方法时调整）
"""

import mujoco  # MuJoCo物理引擎的官方Python绑定
import numpy as np  # 用于数学计算和向量操作
import time  # 用于时间控制和进度计算
from typing import Optional  # 用于类型提示


class RobotWanderer:
    """
    机器人游荡控制器类
    
    该类用于控制机器人在图书馆内随机游荡，遇到障碍物时向左旋转120度继续运动。
    使用MuJoCo物理引擎进行模拟和控制。
    """
    
    def __init__(self, model_path, speed=0.5, turn_angle=2*np.pi/3, collision_threshold=0.1, move_duration=1.0):
        """
        初始化机器人控制器
        
        参数：
            model_path (str): mujoco模型文件路径
            speed (float, optional): 移动速度，默认0.5
            turn_angle (float, optional): 转向角度（弧度），默认120度
            collision_threshold (float, optional): 碰撞检测阈值，默认0.1
            move_duration (float, optional): 每次移动持续时间（秒），默认2秒
        """
        # 初始化核心参数
        self.speed = speed  # 移动速度
        self.turn_angle = turn_angle  # 转向角度（左旋转120度）
        self.collision_threshold = collision_threshold  # 碰撞检测阈值
        self.move_duration = move_duration  # 每次移动持续时间
        self.current_direction = np.array([0, -1, 0])  # 初始方向：向图书馆内部
        self.viewer = None  # 可视化窗口实例

        try:
            # 加载模型（使用新的mujoco库API）
            self.model = mujoco.MjModel.from_xml_path(model_path)  # 加载XML模型文件
            self.data = mujoco.MjData(self.model)  # 创建数据对象
            print("模型加载成功")
        except Exception as e:
            # 模型加载失败时抛出异常
            raise RuntimeError(f"模型加载失败: {str(e)}")

        # 查找电机ID
        self.motor_id = -1  # 初始化电机ID为-1
        print(f"执行器数量: {self.model.nu}")  # 打印执行器数量
        
        # 在新的mujoco库中，需要遍历执行器来查找名称
        for i in range(self.model.nu):
            # 获取执行器名称
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  执行器 {i}: {actuator_name}")
            if actuator_name == 'robot_move_motor':
                self.motor_id = i  # 找到目标电机，记录ID
                break
        
        # 未找到电机时抛出异常
        if self.motor_id == -1:
            raise KeyError("未找到电机'robot_move_motor'，请检查模型文件中的执行器名称")
        print(f"电机ID: {self.motor_id}")

        # 缓存机器人主体ID
        try:
            # 使用mj_name2id查找物体ID
            self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot_vacuum')
            if self.robot_body_id == -1:
                raise KeyError("未找到机器人主体'robot_vacuum'，请检查模型文件中的body名称")
            print("机器人控制器初始化完成")
        except Exception as e:
            # 打印所有物体名称，以便调试
            print("所有物体名称:")
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {i}: {body_name}")
            raise KeyError(f"未找到机器人主体'robot_vacuum'。错误: {str(e)}")

    def start_viewer(self):
        """
        启动可视化窗口
        
        使用MuJoCo的viewer模块启动被动可视化窗口，用于实时查看机器人运动状态。
        """
        import mujoco.viewer  # 延迟导入，减少启动时间
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)  # 启动被动可视化窗口
        print("可视化窗口已启动")

    def check_collision(self):
        """
        检测碰撞
        
        遍历所有几何体，计算其与机器人的距离，判断是否接近障碍物。
        
        返回：
            bool: 是否检测到有效碰撞
        """
        # 获取机器人位置
        robot_pos = self.data.body(self.robot_body_id).xpos
        print(f"【碰撞检测】机器人位置: {robot_pos.round(3)}")
        
        # 无效几何体列表（无需检测碰撞）
        ignore_geoms = ['floor', 'ground', 'wall_invisible', 'table_leg_invisible']

        # 遍历所有几何体检测距离
        min_distance = float('inf')
        closest_geom = ""
        for i in range(self.model.ngeom):
            # 获取几何体名称
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            
            # 跳过机器人自身和无效几何体
            if 'robot_' in geom_name or any(ig in geom_name.lower() for ig in ignore_geoms):
                continue

            # 计算机器人与几何体的距离
            geom_pos = self.data.geom(i).xpos
            distance = np.linalg.norm(geom_pos - robot_pos)
            
            # 记录最近的障碍物
            if distance < min_distance:
                min_distance = distance
                closest_geom = geom_name
            
            if distance < self.collision_threshold:
                print(f"【碰撞检测】距离障碍物 {geom_name} 的距离: {distance:.4f}m，触发避障")
                return True
        
        # 打印最近的障碍物信息
        if closest_geom:
            print(f"【碰撞检测】最近的障碍物: {closest_geom}，距离: {min_distance:.4f}m")
        return False

    def change_direction(self):
        """
        向左旋转固定角度（默认120度）更新方向
        
        基于当前朝向，通过Z轴旋转矩阵计算左旋转后的新方向向量（平面内逆时针旋转）
        """
        # 定义Z轴旋转矩阵（左旋转=逆时针旋转，遵循右手定则）
        cos_theta = np.cos(self.turn_angle)
        sin_theta = np.sin(self.turn_angle)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ])
        
        # 应用旋转矩阵计算新方向
        self.current_direction = np.dot(rotation_matrix, self.current_direction)
        # 归一化方向向量（避免浮点误差累积）
        self.current_direction = self.current_direction / np.linalg.norm(self.current_direction)
        
        # 计算新方向的角度（便于日志输出）
        angle_rad = np.arctan2(self.current_direction[1], self.current_direction[0])
        angle_deg = np.degrees(angle_rad)
        print(f"【方向更新】新方向向量: {self.current_direction[:2].round(3)} | 角度: {angle_deg:.1f}°")

    def move_robot(self, duration):
        """
        控制机器人移动指定时间
        
        参数：
            duration (float): 单次移动持续时间（秒）
        """
        start_time = time.time()  # 记录开始时间
        
        # 在指定时间内持续移动
        while time.time() - start_time < duration:
            # 碰撞检测：发现则执行避障动作
            if self.check_collision():
                print("【避障动作】执行避障：向后移动0.1，左旋转120度")
                
                # 1. 向后移动0.5（增加距离，确保远离障碍物）
                self.move_backward(0.5)
                
                # 2. 右旋转120度
                self.rotate_right(120)
                
                # 3. 额外向前移动一小段距离，确保远离障碍物
                print("【避障动作】避障完成，向新方向移动一小段距离")
                # 向新方向移动0.5米
                self.move_forward(0.5)
                print("【避障动作】远离障碍物完成，继续正常移动")
                
                # 4. 继续运动：重置开始时间，继续向新方向移动
                start_time = time.time()  # 重置开始时间，继续移动
                
                # 立即执行一次模拟步进，应用新的方向
                y_dir = self.current_direction[1]
                control_value = -self.speed if y_dir < 0 else self.speed
                self.data.ctrl[self.motor_id] = control_value
                mujoco.mj_step(self.model, self.data)
                if self.viewer:
                    self.viewer.sync()
                time.sleep(0.1)  # 短暂停顿，确保方向变更生效
                continue

            # 核心更新：根据方向向量确定移动方向
            # 提取Y轴方向分量（适配模型的移动关节），控制移动方向
            y_dir = self.current_direction[1]
            # 提取X轴方向分量，用于判断是否需要改变移动方向
            x_dir = self.current_direction[0]
            
            # 根据方向向量的Y分量确定移动方向，同时考虑X分量
            # 当X分量绝对值大于Y分量时，我们认为机器人应该改变移动方向
            if abs(x_dir) > abs(y_dir):
                # 如果X分量为正，机器人应该向右移动（相当于Y负方向）
                # 如果X分量为负，机器人应该向左移动（相当于Y正方向）
                control_value = -self.speed if x_dir > 0 else self.speed
            else:
                # 否则，根据Y分量的符号确定移动方向
                control_value = -self.speed if y_dir < 0 else self.speed
            
            self.data.ctrl[self.motor_id] = control_value  # 使用计算出的控制值
            
            # 打印关节位置和限制范围
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'robot_move')
            joint_pos = self.data.qpos[joint_id]
            joint_range = self.model.jnt_range[joint_id]
            print(f"【移动控制】方向向量: {self.current_direction[:2].round(3)}, X分量: {x_dir:.3f}, Y分量: {y_dir:.3f}, 控制值: {control_value:.3f}")
            print(f"【关节状态】关节位置: {joint_pos:.4f}, 关节限制: {joint_range}")
            
            # 检测关节是否到达极限位置（位置不再变化）
            if hasattr(self, 'prev_joint_pos'):
                if abs(joint_pos - self.prev_joint_pos) < 1e-4:
                    print("【关节状态】关节已到达极限位置，触发避障动作")
                    # 执行避障动作
                    self.move_backward(0.5)  # 向后移动0.5米，增加距离以更有效地避开障碍物
                    self.rotate_right(120)  # 向右旋转120度，增加角度以更明显地改变方向
                    # 额外向前移动一小段距离，确保远离障碍物
                    print("【避障动作】避障完成，向新方向移动一小段距离")
                    # 向新方向移动0.5米
                    self.move_forward(0.5)
                    print("【避障动作】远离障碍物完成，继续正常移动")
                    start_time = time.time()  # 重置开始时间，继续移动
                    # 立即执行一次模拟步进，应用新的方向
                    y_dir = self.current_direction[1]
                    x_dir = self.current_direction[0]
                    # 根据方向向量的Y分量确定移动方向，同时考虑X分量
                    if abs(x_dir) > abs(y_dir):
                        control_value = -self.speed if x_dir > 0 else self.speed
                    else:
                        control_value = -self.speed if y_dir < 0 else self.speed
                    self.data.ctrl[self.motor_id] = control_value
                    mujoco.mj_step(self.model, self.data)
                    if self.viewer:
                        self.viewer.sync()
                    time.sleep(0.1)  # 短暂停顿，确保方向变更生效
                    continue
            # 保存当前关节位置
            self.prev_joint_pos = joint_pos

            # 步进模拟+可视化刷新
            mujoco.mj_step(self.model, self.data)  # 执行一次模拟步进
            if self.viewer:
                self.viewer.sync()  # 同步可视化窗口
            time.sleep(0.005)  # 微延迟，提高流畅度

        # 移动结束后停止电机（避免惯性滑动）
        self.data.ctrl[self.motor_id] = 0  # 重置控制值为0
        mujoco.mj_step(self.model, self.data)  # 执行一次模拟步进，应用停止命令
        
    def move_backward(self, distance):
        """
        控制机器人向后移动指定距离
        
        参数：
            distance (float): 向后移动的距离
        """
        print(f"【避障动作】开始向后移动，距离：{distance}")
        # 计算移动时间（基于速度和距离）
        move_time = max(distance / self.speed, 0.1)
        start_time = time.time()
        print(f"【避障动作】移动时间：{move_time:.3f}秒，速度：{self.speed}")
        
        # 向后移动
        while time.time() - start_time < move_time:
            # 向后移动（正方向）
            self.data.ctrl[self.motor_id] = self.speed
            print(f"【避障动作】向后移动，控制值：{self.speed}")
            
            # 步进模拟+可视化刷新
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
            time.sleep(0.005)
        
        # 停止电机
        self.data.ctrl[self.motor_id] = 0
        mujoco.mj_step(self.model, self.data)
        print(f"【避障动作】向后移动完成，距离：{distance}")
        
    def move_forward(self, distance):
        """
        控制机器人向前移动指定距离
        
        参数：
            distance (float): 向前移动的距离
        """
        print(f"【避障动作】开始向前移动，距离：{distance}")
        # 计算移动时间（基于速度和距离）
        move_time = max(distance / self.speed, 0.1)
        start_time = time.time()
        print(f"【避障动作】移动时间：{move_time:.3f}秒，速度：{self.speed}")
        
        # 向前移动
        while time.time() - start_time < move_time:
            # 向前移动（根据当前方向）
            y_dir = self.current_direction[1]
            control_value = -self.speed if y_dir < 0 else self.speed
            self.data.ctrl[self.motor_id] = control_value
            print(f"【避障动作】向前移动，控制值：{control_value}")
            
            # 步进模拟+可视化刷新
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()
            time.sleep(0.005)
        
        # 停止电机
        self.data.ctrl[self.motor_id] = 0
        mujoco.mj_step(self.model, self.data)
        print(f"【避障动作】向前移动完成，距离：{distance}")
        
    def rotate_left(self, angle_deg):
        """
        控制机器人左旋转指定角度
        
        参数：
            angle_deg (float): 左旋转的角度（度）
        """
        print(f"【避障动作】开始左旋转，角度：{angle_deg}°")
        # 将角度转换为弧度
        angle_rad = np.radians(angle_deg)
        print(f"【避障动作】旋转角度（弧度）：{angle_rad:.3f}")
        
        # 计算当前方向的角度
        current_angle = np.arctan2(self.current_direction[1], self.current_direction[0])
        print(f"【避障动作】当前方向角度（弧度）：{current_angle:.3f}，（度）：{np.degrees(current_angle):.1f}")
        
        # 计算新方向的角度（左旋转：逆时针旋转，减去角度）
        new_angle = current_angle - angle_rad
        print(f"【避障动作】新方向角度（弧度）：{new_angle:.3f}，（度）：{np.degrees(new_angle):.1f}")
        
        # 计算新的方向向量
        self.current_direction = np.array([np.cos(new_angle), np.sin(new_angle), 0])
        print(f"【避障动作】左旋转完成，角度：{angle_deg}°，新方向向量：{self.current_direction[:2].round(3)}")
    
    def rotate_right(self, angle_deg):
        """
        控制机器人右旋转指定角度
        
        参数：
            angle_deg (float): 右旋转的角度（度）
        """
        print(f"【避障动作】开始右旋转，角度：{angle_deg}°")
        # 将角度转换为弧度
        angle_rad = np.radians(angle_deg)
        print(f"【避障动作】旋转角度（弧度）：{angle_rad:.3f}")
        
        # 计算当前方向的角度
        current_angle = np.arctan2(self.current_direction[1], self.current_direction[0])
        print(f"【避障动作】当前方向角度（弧度）：{current_angle:.3f}，（度）：{np.degrees(current_angle):.1f}")
        
        # 计算新方向的角度（右旋转：顺时针旋转，加上角度）
        new_angle = current_angle + angle_rad
        print(f"【避障动作】新方向角度（弧度）：{new_angle:.3f}，（度）：{np.degrees(new_angle):.1f}")
        
        # 计算新的方向向量
        self.current_direction = np.array([np.cos(new_angle), np.sin(new_angle), 0])
        print(f"【避障动作】右旋转完成，角度：{angle_deg}°，新方向向量：{self.current_direction[:2].round(3)}")

    def wander(self, duration=60):
        """
        让机器人在图书馆内游荡指定时间
        
        参数：
            duration (float, optional): 总游荡时间（秒），默认60秒
        """
        print(f"\n【开始游荡】总时间: {duration}秒 | 单次移动: {self.move_duration}秒 | 左旋转角度: {np.degrees(self.turn_angle):.0f}度")
        start_time = time.time()  # 记录开始时间

        # 在指定时间内持续游荡
        while time.time() - start_time < duration:
            # 计算剩余时间，防止最后一次移动超时
            remaining_time = duration - (time.time() - start_time)
            current_move_duration = min(self.move_duration, remaining_time)

            # 执行单次移动
            self.move_robot(current_move_duration)

            # 输出进度
            elapsed_time = time.time() - start_time
            progress = (elapsed_time / duration) * 100
            print(f"【游荡进度】已运行 {elapsed_time:.1f}秒 | 剩余 {remaining_time:.1f}秒 | 完成 {progress:.1f}%\n")

            # 转向后短暂停顿，避免操作过快
            time.sleep(0.3)

        print(f"【游荡结束】实际运行时间: {time.time() - start_time:.1f}秒")

    def close(self):
        """
        关闭模拟
        
        停止电机并关闭可视化窗口，确保模拟正常结束。
        """
        if self.viewer:
            self.data.ctrl[self.motor_id] = 0  # 停止电机
            mujoco.mj_step(self.model, self.data)  # 执行一次模拟步进，应用停止命令
            self.viewer.close()  # 关闭可视化窗口
            print("\n【模拟关闭】可视化窗口已关闭，电机已停止")


if __name__ == "__main__":
    # 模型路径
    model_path = "bookstore-lounge.xml"
    # 初始化机器人（支持自定义参数）
    robot = None
    
    try:
        # 自定义参数示例：调整速度、碰撞阈值，可自定义转向角度（如90度=np.pi/2）
        robot = RobotWanderer(
            model_path=model_path,
            speed=3.0,               # 移动速度（已调快）
            collision_threshold=0.2, # 碰撞检测阈值（减小到0.2米，更容易触发避障）
            # turn_angle=2*np.pi/3      # 如需修改转向角度，取消注释并设置（示例：120度）
        )
        robot.start_viewer()          # 启动可视化窗口
        robot.wander(duration=120)    # 游荡2分钟
    except Exception as e:
        # 捕获并打印运行错误
        print(f"\n【运行错误】{type(e).__name__}: {str(e)}")
    finally:
        # if __name__ == "__main__":确保模拟正常关闭
        if robot is not None:
            robot.close()
