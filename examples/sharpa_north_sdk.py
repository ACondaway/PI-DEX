"""Reference Sharpa North live SDK (Zenoh + protobuf) snapshot.

This file is an external-env reference for PI-DEX handoff phase 6 hardware work.
It is not importable as a first-party package module: it depends on sibling
``.base`` / ``.builder`` / ``.proto`` from the vendor tree.

Observation and action key identity used by PI-DEX lives in
``pi_dex.robot.sharpa_runtime_keys``. A future ``BimanualController`` adapter must
implement atomic bimanual apply/hold/lease semantics from
``pi_dex.serve.deployment``; the paced ``_action_sender_loop`` below is not that
protocol.
"""

import threading
import time
import numpy as np
import zenoh
import json
from turbojpeg import TurboJPEG
from .base import Env
from .builder import Environment
from .proto import north_pb2 as pb

@Environment.register_module()
class NorthDirect(Env):
    """    
    发送: UhrActionBundle (动作数据)
    接收: NorthObservation (观察数据)
    使用Zenoh进行通信
    """
    
    def __init__(self, 
                 observation_topic="north_observation",
                 action_topic="inference/action",
                 action_pub_duration=0.01666,
                 zenoh_config=None,
                 **kwargs):
        super().__init__(**kwargs, using_zmq=False)
        self.logger.info(f"{kwargs}")
        self.action_pub_key = kwargs.get("action_output", [])

        self.observation_topic = observation_topic
        self.action_topic = action_topic
        self.action_pub_duration = action_pub_duration

        # Zenoh相关
        self.zenoh_session = None
        self.obs_subscriber = None
        self.action_publisher = None
        self.zenoh_config = zenoh_config or {}
        
        # 状态管理
        self.last_observation = None
        self.is_connected = False
        self.turbo_jpeg = TurboJPEG()
        
        # 初始化Zenoh连接
        self._init_zenoh_connections()

        # 线程控制相关
        self._action_thread_stop_event = threading.Event()
        self._action_thread = None
        self._action_lock = threading.Lock()
        
        # 动作缓冲区
        self.output_text = None
        self._action_buffer = []
        
        # 启动常驻发送线程
        self._start_action_thread()

    def _init_zenoh_connections(self):
        """初始化Zenoh连接"""
        try:
            # 创建Zenoh配置
            conf = zenoh.Config()
            
            # 应用用户配置
            if self.zenoh_config:
                for key, value in self.zenoh_config.items():
                    conf.insert_json5(key, json.dumps(value))
            
            # 创建Zenoh会话
            self.zenoh_session = zenoh.open(conf)
            
            # 创建观察数据订阅者
            self.obs_subscriber = self.zenoh_session.declare_subscriber(
                self.observation_topic,
                self._on_observation_received
            )
            
            # 创建动作发布者
            self.action_publisher = self.zenoh_session.declare_publisher(self.action_topic)
            
            self.is_connected = True
            self.logger.info(f"Zenoh initialized - obs: {self.observation_topic}, action: {self.action_topic}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Zenoh connections: {e}")
            self.is_connected = False
            return False

    def _on_observation_received(self, sample):
        """Zenoh回调：接收观察数据"""
        try:
            # 解析NorthObservation消息
            north_obs = pb.NorthObservation()
            north_obs.ParseFromString(sample.payload)
            
            # 转换为字典格式
            obs_dict = self._convert_north_observation_to_dict(north_obs)
            
            # 更新最新观察数据
            self.last_observation = obs_dict
            
        except Exception as e:
            self.logger.error(f"Failed to receive observation: {e}")

    def _convert_north_observation_to_dict(self, north_obs):
        """将NorthObservation protobuf消息转换为字典格式"""
        obs_dict = {
            "timestamp": north_obs.timestamp,
            "reward": north_obs.reward,
            "on_sleep": north_obs.on_sleep,
        }
        
        # 处理机器人状态数据
        if north_obs.HasField('robot_state'):
            robot_state = north_obs.robot_state
            
            # 左臂状态
            if robot_state.HasField('left_arm'):
                left_arm = robot_state.left_arm
                obs_dict['/state/left_arm/joint_angle'] = list(left_arm.joint.position)
                force = [left_arm.wrench.force.x, left_arm.wrench.force.y, left_arm.wrench.force.z]
                torque = [left_arm.wrench.torque.x, left_arm.wrench.torque.y, left_arm.wrench.torque.z]
                obs_dict['/state/left_arm/tcp_forces'] = force + torque
                
            # 右臂状态
            if robot_state.HasField('right_arm'):
                right_arm = robot_state.right_arm
                obs_dict['/state/right_arm/joint_angle'] = list(right_arm.joint.position)
                force = [right_arm.wrench.force.x, right_arm.wrench.force.y, right_arm.wrench.force.z]
                torque = [right_arm.wrench.torque.x, right_arm.wrench.torque.y, right_arm.wrench.torque.z]
                obs_dict['/state/right_arm/tcp_forces'] = force + torque
                
            # 左手状态
            if robot_state.HasField('left_hand'):
                left_hand = robot_state.left_hand
                obs_dict['/state/left_hand/joint_angle'] = list(left_hand.joint.position)
                obs_dict['/state/left_hand/effort'] = list(left_hand.joint.effort)
                
            # 右手状态
            if robot_state.HasField('right_hand'):
                right_hand = robot_state.right_hand
                obs_dict['/state/right_hand/joint_angle'] = list(right_hand.joint.position)
                obs_dict['/state/right_hand/effort'] = list(right_hand.joint.effort)
            # 电机状态
            if robot_state.HasField('motor'):
                motor_status = robot_state.motor
                motor_positions = [motor.position for motor in motor_status.motors]
                motor_velocities = [motor.velocity for motor in motor_status.motors]
                motor_torques = [motor.torque for motor in motor_status.motors]
                obs_dict['/state/motor/joint_angle'] = motor_positions
                obs_dict['/state/motor/joint_velocity'] = motor_velocities
                obs_dict['/state/motor/joint_effort'] = motor_torques
        
        # 处理视觉数据
        if north_obs.HasField('vision'):
            vision = north_obs.vision
            
            if vision.HasField('image_left'):
                encoded_image = np.frombuffer(vision.image_left.data, np.uint8)
                image = self.turbo_jpeg.decode(encoded_image)
                obs_dict["/observe/vision/head/stereo/lefteye/rgb"] = image[:, :, ::-1]
                
            if vision.HasField('image_right'):
                encoded_image = np.frombuffer(vision.image_right.data, np.uint8)
                image = self.turbo_jpeg.decode(encoded_image)
                obs_dict["/observe/vision/head/stereo/righteye/rgb"] = image[:, :, ::-1]
                
            if vision.HasField('fish_left'):
                encoded_image = np.frombuffer(vision.fish_left.data, np.uint8)
                image = self.turbo_jpeg.decode(encoded_image)
                obs_dict["/observe/vision/left_wrist/fisheye/rgb"] = image[:, :, ::-1]
                
            if vision.HasField('fish_right'):
                encoded_image = np.frombuffer(vision.fish_right.data, np.uint8)
                image = self.turbo_jpeg.decode(encoded_image)
                obs_dict["/observe/vision/right_wrist/fisheye/rgb"] = image[:, :, ::-1]
        
        # 处理模式信息
        if north_obs.HasField('mode'):
            mode = north_obs.mode
            obs_dict["mode"] = {
                "operation_mode": mode.operation_mode,
                "state": mode.state,
                "sub_state": mode.sub_state
            }
            obs_dict["/mode/act"] = mode.state
            obs_dict["/mode/sub_act"] = mode.sub_state
            
        obs_dict['/task_code'] = 0
        
        if north_obs.HasField('language'):
            obs_dict["/language"] = north_obs.language

        return obs_dict

    def _start_action_thread(self):
        """启动常驻动作发送线程"""
        self._action_thread_stop_event.clear()
        self._action_thread = threading.Thread(
            target=self._action_sender_loop,
            daemon=True
        )
        self._action_thread.start()

    def _action_sender_loop(self):
        """动作发送循环"""
        last_send_time = time.time() - self.action_pub_duration
        
        while not self._action_thread_stop_event.is_set():
            current_time = time.time()
            time_since_last = current_time - last_send_time
            
            # 如果还没到发送间隔，sleep到间隔时间
            if time_since_last < self.action_pub_duration:
                sleep_time = last_send_time + self.action_pub_duration - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue
            
            # 获取并发送下一个动作
            with self._action_lock:
                if not self._action_buffer:
                    last_send_time = time.time()
                    continue
                
                raw_action = self._action_buffer.pop(0)
                action_dict = self._prepare_action_dict(raw_action)
            
            self._send_action(action_dict)
            last_send_time = time.time()

    def _prepare_action_dict(self, action_values):
        """将原始数据格式转换为发送用的动作字典"""
        action_dict = {}
        for key in self.action_pub_key:
            parts = key.split('/')
            if len(parts) < 4:
                continue
            actuator = parts[2]
            action_type = parts[3]
            
            if key in action_values:
                if actuator not in action_dict:
                    action_dict[actuator] = {}
                action_dict[actuator]["position"] = action_values[key]
        
        return action_dict

    def _send_actions(self, actions_dict):
        """更新动作缓冲区（转换为List[Dict]格式）"""
        with self._action_lock:
            # 获取所有key的列表长度
            chunk_size = None
            for key, values in actions_dict.items():
                if "/action" not in key:
                    continue
                if chunk_size is None:
                    chunk_size = len(values)
                if len(values) != chunk_size:
                    raise ValueError("All action lists must have the same length")
            
            # 转换为List[Dict]格式
            if chunk_size > 0:
                self._action_buffer = [
                    {k: values[i] for k, values in actions_dict.items() if '/action' in k}
                    for i in range(chunk_size)
                ]
            else:
                self._action_buffer = []

            if "output_text" in actions_dict:
                self.output_text = actions_dict["output_text"]

    def _send_action(self, action_dict):
        """发送动作数据（使用Zenoh）"""
        if not self.is_connected or not self.action_publisher:
            self.logger.warning("Zenoh not connected, cannot send action")
            return False
            
        try:
            # 创建UhrActionBundle消息
            action_bundle = pb.UhrActionBundle()
            
            # 设置时间戳
            current_time = time.time()
            
            # 处理左手套动作
            if "left_hand" in action_dict:
                left_glove = action_bundle.left_glove
                self._fill_joint_data(left_glove.joint, action_dict["left_hand"])
                self._set_header_timestamp(left_glove.header, current_time)
                
            # 处理右手套动作
            if "right_hand" in action_dict:
                right_glove = action_bundle.right_glove
                self._fill_joint_data(right_glove.joint, action_dict["right_hand"])
                self._set_header_timestamp(right_glove.header, current_time)
                
            # 处理左臂动作
            if "left_arm" in action_dict:
                left_arm = action_bundle.left_arm
                self._fill_joint_data(left_arm.joint, action_dict["left_arm"])
                self._set_header_timestamp(left_arm.header, current_time)
                
            # 处理右臂动作
            if "right_arm" in action_dict:
                right_arm = action_bundle.right_arm
                self._fill_joint_data(right_arm.joint, action_dict["right_arm"])
                self._set_header_timestamp(right_arm.header, current_time)
            
            # 处理电机动作
            if "motor" in action_dict:
                motor = action_bundle.motor
                motor.timestamp = int(current_time * 1000)
                self._fill_motor_commands(motor, action_dict["motor"])
            
            if self.output_text is not None:
                action_bundle.language = self.output_text
            
            # 序列化并通过Zenoh发送
            serialized_data = action_bundle.SerializeAsString()
            self.action_publisher.put(serialized_data)

            self.logger.debug(f"Sent action bundle with {len(serialized_data)} bytes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send action: {e}")
            return False

    def _fill_joint_data(self, joint_proto, joint_dict):
        """填充关节数据到protobuf消息"""
        if "position" in joint_dict:
            joint_proto.position[:] = joint_dict["position"]
        if "velocity" in joint_dict:
            joint_proto.velocity[:] = joint_dict["velocity"]
        if "effort" in joint_dict:
            joint_proto.effort[:] = joint_dict["effort"]
        if "name" in joint_dict:
            joint_proto.name[:] = joint_dict["name"]

    def _fill_motor_commands(self, motor_proto, motor_dict):
        """填充电机命令到protobuf消息"""
        if "position" in motor_dict:
            positions = motor_dict["position"]
            for i, pos in enumerate(positions):
                motor_cmd = motor_proto.commands.add()
                motor_cmd.motor_id = i + 1
                motor_cmd.command = "position"
                motor_cmd.value = pos

    def _set_header_timestamp(self, header, timestamp):
        """设置header时间戳"""
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec

    def send_actions(self, actions_dict):
        """发送动作数据（公开接口）"""
        self._send_actions(actions_dict)

    def get_last_observation(self):
        """获取最新的观察数据（公开接口）"""
        return self.last_observation

    def shutdown(self):
        """停止线程和清理资源"""
        self._action_thread_stop_event.set()
        if self._action_thread is not None:
            self._action_thread.join()
        self.cleanup()

    def cleanup(self):
        """清理Zenoh资源"""
        try:
            if self.obs_subscriber:
                self.obs_subscriber.undeclare()
            if self.action_publisher:
                self.action_publisher.undeclare()
            if self.zenoh_session:
                self.zenoh_session.close()
            self.is_connected = False
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """析构函数，清理资源"""
        self.cleanup()
