import numpy as np
import gym
import numpy.random as random
import matplotlib.pyplot as plt



MAX_EPOTCH = 2000
env = gym.make('CartPole-v0')
w_size = env.observation_space.shape[0]
w = random.random(w_size)

epoch_reward = 0.
all_reward = [0]
for ep in range(MAX_EPOTCH):
    observation = env.reset()
    candidate_weight = random.random(w_size) * 0.01 + w
    for t in range(1000):
        # env.render()
        weighted_sum = np.dot(candidate_weight, observation)
        if weighted_sum >= 0:
            action = 1 
        else:
            action = 0 
    
        observation, reward, done, info = env.step(action)
        epoch_reward += reward
        if done:
            ### if the new policy performs better, update the weight
            if epoch_reward < 150 and epoch_reward > all_reward[-1]:
                w = candidate_weight
            all_reward.append(epoch_reward)
            print('epoch: ', ep+1 , ', total reward: ', epoch_reward)
            epoch_reward = 0
            break
plt.title('CartPole-v0')
plt.plot(all_reward)
plt.show()