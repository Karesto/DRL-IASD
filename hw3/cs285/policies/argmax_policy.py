import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## DONE return the action that maxinmizes the Q-value
        # at the current observation as the output
        action = self.critic.qa_values(observation).argmax(axis=1)
        return action.squeeze()

class EpsGreedyPolicy(object):

    def __init__(self, critic, eps, att):
        self.critic = critic
        self.eps = eps
        self.att = att

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## DONE return the action that maxinmizes the Q-value
        # at the current observation as the output
        prob = np.random.binomial(1, np.eps, 1)[0]
        totqa = self.critic.qa_values(observation)
        if prob < self.eps:
            action = np.random.randint(totqa.shape[1])
            return action
        else:
            action = totqa.argmax(axis=1)
            return action.squeeze()

    def attenuate(self):
        self.eps *= self.att