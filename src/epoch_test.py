import numpy as np
import gym

'''
test an epoch of CartPole-v0 with given weight
'''

def evaluate_given_parameter_by_sign(env, weight):
    '''
    Args:
        env (gym.env): the environment of gym
        weight (numpy array): a matrix in dimention: observation_dim x action_dim
    '''
    observation = env.reset()
    total_reward = 0.

    for t in range(1000):

        env.render()
        weighted_sum = np.dot(weight, observation)
        if weighted_sum >= 0:
            action = 1 
        else:
            action = 0 
    
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

env = gym.make('CartPole-v0')
w = np.random.random(4)
r = evaluate_given_parameter_by_sign(env, w)
print('epoch reward: ', r)