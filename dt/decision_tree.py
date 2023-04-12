#!/usr/bin/python3
"""
Implementation of the abstract class BehaviourTree

Author: Leonardo Lucio Custode
Creation Date: 08-04-2020
Last modified: mer 6 mag 2020, 11:28:42
"""
import abc
import numpy as np


class DecisionTree:
    def __init__(self):
        self.current_reward = 0
        self.last_leaf = None

    @abc.abstractmethod
    def get_action(self, input):
        pass
    
    def set_reward(self, reward):
        self.current_reward = reward

    def new_episode(self):
        self.last_leaf = None


class Leaf:
    def get_action(self):
        pass

    def update(self, x):
        pass


class QLearningLeaf(Leaf):
    def __init__(self, n_actions, learning_rate, discount_factor):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.parent = None
        self.iteration = [1] * n_actions

        self.q = np.zeros(n_actions, dtype=np.float32)
        self.last_action = 0

    def get_action(self):
        action = np.argmax(self.q)
        self.last_action = action
        return action

    def update(self, reward, qprime):
        if self.last_action is not None:
            lr = self.learning_rate if not callable(self.learning_rate) else self.learning_rate(self.iteration[self.last_action])
            if lr == "auto":
                lr = 1/self.iteration[self.last_action]
            self.q[self.last_action] += lr * (
                        reward + self.discount_factor * qprime - self.q[self.last_action])

    def next_iteration(self):
        self.iteration[self.last_action] += 1

    def __repr__(self):
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        return repr(self)


class EpsGreedyLeaf(QLearningLeaf):
    def __init__(self, n_actions, learning_rate, discount_factor, epsilon):
        super().__init__(n_actions, learning_rate, discount_factor)
        self.epsilon = epsilon

    def get_action(self):
        # Epsilon: random action
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Get the argmax. If there are equal values, choose randomly between them
            best = [None]
            max_ = -float("inf")
            
            for i, v in enumerate(self.q):
                if v > max_:
                    max_ = v
                    best = [i]
                elif v == max_:
                    best.append(i)

            action = np.random.choice(best)

        self.last_action = action
        self.next_iteration()
        return action

class RandomlyInitializedEpsGreedyLeaf(EpsGreedyLeaf):
    def __init__(self, n_actions, learning_rate, discount_factor, epsilon, low=-100, up=100):
        """
        Initialize the leaf.
        Params:
            - n_actions: The number of actions
            - learning_rate: the learning rate to use, callable or float
            - discount_factor: the discount factor, float
            - epsilon: epsilon parameter for the random choice
            - low: lower bound for the initialization
            - up: upper bound for the initialization
        """
        super(RandomlyInitializedEpsGreedyLeaf, self).__init__(n_actions, learning_rate, discount_factor, epsilon)
        self.q = np.random.uniform(low, up, n_actions)
