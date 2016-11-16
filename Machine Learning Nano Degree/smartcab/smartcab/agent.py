# coding: utf-8
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # location和heading??
        # self.state = None
        
        # TODO: Select action according to your policy
        action = self.simple_action(inputs)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, next_waypont = {}, inputs = {}, action = {}, reward = {}"\
            .format(deadline, self.next_waypoint, inputs, action, reward)  # [debug]

    def simple_action(self, inputs):
        """
        不以destination为目标，仅仅从当前的inputs判断哪个方向可以通行
        :param inputs:
        :return:
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

    def add_env(self, env):
        assert isinstance(env, Environment), 'TypeError: invalid env type %s' % type(env)
        self.env = env


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=0)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.add_env(e)
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
