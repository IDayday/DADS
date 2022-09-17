import gym
from soft_actor_critic.environments.environment import Environment
import numpy as np
from gym.spaces import Box

'''
Ant环境：状态有111个，前27个有值，包括蚂蚁身体不同部位的位置值，接着是这些单独部位的速度(它们的导数)
，后86=14*6个是施加到每个连杆质心的接触力，这14个环节是:地面环节、躯干环节、每条腿的3个环节(1 + 1 + 12)
和6个外力，在v3版本和v2版本接触力都为0。
前27个状态介绍：
    |0|躯干的z坐标|-inf,inf|单位：m                     |1|躯干的x方向|-inf,inf|单位：rad
    |2|躯干的y方向|-inf,inf|单位：rad                   |3|躯干的z方向|-inf,inf|单位：rad
    |4|w-躯干方向 |-inf,inf|单位：rad                   |5|躯干和左前方第一个连杆之间的角度|-inf,inf|单位：rad
    |6|左前方两个连杆之间的角度|-inf,inf|单位：rad       |7|躯干和右前侧第一个连杆之间的角度|-inf,inf|单位：rad
    |8|右前侧两个连杆之间的角度|-inf,inf|单位：rad       |9|躯干和左后侧第一个链接之间的角度|-inf,inf|单位：rad
    |10|左后侧两个链接之间的角度|-inf,inf|单位：rad      |11|躯干和右后侧第一个环节之间的角度|-inf,inf|单位：rad
    |12|右后侧两个链接之间的角度|-inf,inf|单位：rad      |13|躯干的x坐标速度|-inf,inf|单位：m/s
    |14|躯干的y坐标速度|-inf,inf|单位：m/s               |15|躯干的z坐标速度|-inf,inf|单位：m/s
    |16|躯干的x坐标角速度|-inf,inf|单位：rad/s           |17|躯干的y坐标角速度|-inf,inf|单位：rad/s
    |18|躯干的z坐标角速度|-inf,inf|单位：rad/s           |19|躯干和左前连杆之间角度的角速度|-inf,inf|单位：rad 
    |20|左前连杆之间角度的角速度|-inf,inf|单位：rad      |21|躯干和右前连杆之间角度的角速度|-inf,inf|单位：rad 
    |22|右前连杆之间角度的角速度|-inf,inf|单位：rad      |23|躯干和左后连杆之间角度的角速度|-inf,inf|单位：rad 
    |24|左后连杆之间角度的角速度|-inf,inf|单位：rad      |25|躯干和右后连杆之间角度的角速度|-inf,inf|单位：rad 
    |26|右后连杆之间角度的角速度|-inf,inf|单位：rad   
动作是8个施加在铰链接合处的扭矩，是连续量。
8个动作介绍：
    |0|施加在躯干和左前臀部之间的转子上的扭矩|-1,1|单位：N*m   |1|施加在左前两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |2|施加在躯干和右前臀部之间的转子上的扭矩|-1,1|单位：N*m   |9|施加在右前两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |4|施加在躯干和左后臀部之间的转子上的扭矩|-1,1|单位：N*m   |5|施加在左后两个连杆之间的转子上的扭矩|-1,1|单位：N*m
    |6|施加在躯干和右后臀部之间的转子上的扭矩|-1,1|单位：N*m   |7|施加在右后两个连杆之间的转子上的扭矩|-1,1|单位：N*m
4个奖励介绍：
奖励由四个部分组成：reward=healthy_r+forward_r-contrl_cost-contact_cost
healthy_r:每个时间里蚂蚁是否健康，每健康一秒它得到一个固定值的奖励
forward_r：前进的奖励，如果蚂蚁向前移动(在正x方向)，这个奖励将是正的。
control_cost:如果蚂蚁采取过大的行动，惩罚蚂蚁的负奖励
contact_cost:如果外部接触力过大，惩罚蚂蚁的负奖励  
2个结束介绍：
游戏的回合结束介绍——done：      
1.如果蚂蚁不健康就结束回合（环境自带）
2.如果持续时间达到1000个时间步长就结束

关于奖励的设置还可以参考：https://blog.csdn.net/weixin_38909635/article/details/105601808
'''
class Ant_Truncated_State(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('Ant-v3')
        # self.observation_space = self.env.observation_space
        self.observation_space = Box(-float('inf'), float('inf'), shape=(27,))
        self.action_space = self.env.action_space
        self.xy_coords = self.env.sim.data.qpos.flat[:2]

    def take_action(self, action):
        _, reward, done, _ = self.env.step(action)
        # cutting out the parts of the state space that the DADS paper similarly removed
        obs = np.concatenate([
          self.env.sim.data.qpos.flat[2:15],
          self.env.sim.data.qvel.flat[:14], ])
        self.xy_coords = list(self.env.sim.data.qpos.flat[:2])
        self.observation = list(obs)
        self.reward = reward
        self.done = done
        self.frames += 1
        if self.done and self.frames > 1000:
            self.won = True
        else:
            self.won = False
        return obs, reward, done, self.won

    def reset_env(self):
        self.env.reset()
        self.observation = np.concatenate([
          self.env.sim.data.qpos.flat[2:15],
          self.env.sim.data.qvel.flat[:14], ])
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
        return self.observation
