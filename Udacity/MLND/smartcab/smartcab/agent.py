# coding: utf-8
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class ActionQValue(object):
    def __init__(self, action, q_value):
        self.action, self.q_value = action, q_value


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        if not isinstance(env, Environment):
            raise TypeError('invalid type %s' % type(env))

        self.env, self.state, self.next_waypoint = env, None, None
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_matrix, self.alpha, self.gamma, self.epsilon = dict(), 0.9, 0.3, 0.1
        self.pre_state, self.pre_action, self.pre_reward = None, None, None
        self.default_q = 1
        self.num_success = 0
        self.env.acc = 0.0

    def reset(self, destination=None):
        """
        每一次的episode都需要重置部分变量
        :param destination:
        :return:
        """
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.pre_action, self.pre_state, self.pre_reward = None, None, None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.update_state(inputs, self.next_waypoint)

        # TODO: Select action according to your policy
        # action = self.simple_action(inputs=inputs)
        action_q = self.select_action_q(self.state)
        action, max_q = action_q.action, action_q.q_value
        # Execute action and get reward
        reward = self.env.act(agent=self, action=action)

        if self.pre_state is not None:
            self.q_matrix[(self.pre_state, self.pre_action)] = self.default_q

            # TODO: Learn policy based on state, action, reward
            self.q_matrix[(self.pre_state, self.pre_action)] = \
                (1 - self.alpha) * self.q_matrix[(self.pre_state, self.pre_action)] + \
                self.alpha * (self.pre_reward + self.gamma * self.select_action_q(self.state).q_value)

        self.pre_state = self.state
        self.pre_action = action
        self.pre_reward = reward
        if self.pre_reward == 12:
            self.num_success += 1
            self.env.acc = self.num_success / 100.0

        print "LearningAgent.update(): deadline = {}, state = {}, inputs = {}, action = {}, reward = {}, acc = {:.2f}"\
            .format(deadline, self.state, inputs, action, reward, self.env.acc)  # [debug]

    def select_action_q(self, state):
        """
        获取action和对应的rewards
        需要注意的是：在Q-learning学习的过程中，需要利用Q-matrix选择下一个action
        但是，在训练的初期，Q-matrix还没有获取那么多信息，因此，如果每次从Q-matrix中获取信息，
        那么就会陷入局部结果中，导致最后无法到达目的地。因此，解决的办法是利用概率要么随机选择
        action,要么从Q-matrix中获取action
        :param state:
        :return:
        """
        if self.__random_choice():
            action = random.choice(self.env.valid_actions)
            return ActionQValue(action, self.q_matrix.get((state, action), self.default_q))
        else:
            # 从Q-matrix中选择argmax_a'{Q(s',a')}的action
            try:
                _valid_actions = self.env.valid_actions
                random.shuffle(_valid_actions)
                _max_q = max([self.q_matrix.get((state, _action), self.default_q) for _action in _valid_actions])
                for _action in _valid_actions:
                    if self.q_matrix.get((state, _action), self.default_q) == _max_q:
                        return ActionQValue(_action, _max_q)
            except KeyError, e:
                print(e.message)

    def __random_choice(self):
        return random.random() < self.epsilon

    def update_state(self, inputs, next_waypoint):
        """
        根据目前的inputs来更新目前的state
        :param inputs: light, oncoming, left, right
        :param next_waypoint: 下一步的方向
        :return:
        """
        if not isinstance(inputs, dict):
            raise TypeError('invalid type %s' % type(inputs))
        for _input in self.env.valid_inputs.keys():
            if _input not in inputs:
                raise KeyError('key %s not in inputs' % _input)

        del inputs['right']  # 因为右边的方向对于下一步行为的判断是没有作用的，故删除
        # del inputs['left']
        inputs['next_waitpoint'] = next_waypoint
        return tuple(sorted(inputs.items()))

    def simple_action(self, inputs):
        """
        如果不利用Q-learning，调用该函数也可以获取下一个action
        最终agent可以顺利到达目的地
        :param inputs: light, oncoming, left, right
        :return: action
        """
        action_okay = True
        # 如果往右边走但信号灯是红灯而且左边有车往右边通行，则车子不能动
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        # 如果向前走但前面是红灯，则车子不能动
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        # 如果往左边走有两种情况车子不能走
        # 1. 信号灯是红灯
        # 2. 前面有车直行或右边有往左边行驶的车辆
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            # self.next_waypoint = random.choice(Environment.valid_actions[1:])
            self.next_waypoint = self.planner.next_waypoint()
        return action


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=0)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.5, display=True)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
