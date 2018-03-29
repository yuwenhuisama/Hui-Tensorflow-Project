'''
File: \Game.py
Project: 15-SimpleReinforcementLearning-Value-Based
Created Date: Wednesday March 28th 2018
Author: Huisama
-----
Last Modified: Thursday March 29th 2018 11:10:50 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# %matplotlib inline
# from IPython import get_ipython
# get_ipython().

class GameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class GameEnv():
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        plt.imshow(a, interpolation="nearest")

    def reset(self):
        self.objects = []
        hero = GameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        goal = GameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        hole = GameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        goal2 = GameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        hole2 = GameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        goal3 = GameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        hole3 = GameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole3)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        self.objects[0] = hero

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        # get all the position in game space
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        # get all the positions of current objects
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        # remove positions of current objects
        for pos in currentPositions:
            points.remove(pos)
        # randomly choice a position
        location = np.random.choice(range(len(points)), replace = False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(GameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                return other.reward, False
        return 0.0, False

    def renderEnv(self):
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1 : item.y + item.size + 1, item.x + 1: item.x + item.size + 1, item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp = 'nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp = 'nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp = 'nearest')
        a = np.stack([b, c, d], axis = 2)
        return a

    def step(self, action):
        self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, reward, done