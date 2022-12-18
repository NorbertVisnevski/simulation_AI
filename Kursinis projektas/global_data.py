class HyperParameters:
    learning_rate = 0.001
    inverse_alpha = 1 - learning_rate
    discount = 0.5
    epsilon = 1.0
    epsilon_decay = 0.8
    epoch = 0
    episode = 0
    observation_size = 7
    action_size = 4
    replay_memory_size = 2000

    @staticmethod
    def decay_epsilon():
        HyperParameters.epsilon *= HyperParameters.epsilon_decay
        if HyperParameters.epsilon < 0.01:
            HyperParameters.epsilon = 0


BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
GREEN1 = (0, 150, 0)
GREEN2 = (0, 150, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
FOOD_COLOR = (10, 200, 150)
