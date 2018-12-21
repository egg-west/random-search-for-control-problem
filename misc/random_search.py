import numpy as np
import gym
import numpy.random as random
import matplotlib.pyplot as plt
# hill climb
# TODO: update every fram or every epoch?
MAX_EPOTCH = 1000
env = gym.make('CartPole-v0')
w_size = env.observation_space.shape[0]
w = random.random(w_size)

epoch_reward = 0.
all_reward = []
for ep in range(MAX_EPOTCH):
    observation = env.reset()
    for t in range(1000):

        # env.render()
        candidate_weight = random.random(w_size) + w

        weighted_sum = np.dot(candidate_weight, observation)
        if weighted_sum >= 0:
            action = 1 
        else:
            action = 0 
    
        observation, reward, done, info = env.step(action)
        if reward + epoch_reward > epoch_reward:
            w = candidate_weight
            epoch_reward += reward
        if done:
            all_reward.append(epoch_reward)
            print('epoch: ', ep+1 , ', total reward: ', epoch_reward)
            epoch_reward = 0
            break
plt.title('CartPole-v0')
plt.plot(all_reward)
plt.show()