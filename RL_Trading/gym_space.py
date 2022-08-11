import gym
import numpy as np
from gym import spaces

# space = spaces.Discrete(8)
# x = space.sample()
# print(x)

space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
# x = space.sample()
# print(x)
space1 = spaces.Box(low=-1, high=1, shape=(1,))
y = space1.sample()
print(y*100)
