class HyperParameters:
    learning_rate = 0.001
    inverse_alpha = 1 - learning_rate
    discount = 0.5
    epsilon = 0.1
    epsilon_decay = 0.9
    epoch = 0
    observation_size = 7
    action_size = 4
    replay_memory_size = 2000

    _episode = 7345
    episode = _episode + 1
    learning = False
    deep_learning = False
    AI_DIRECTORY = "Q_LEARNING_ALPHA02_NO_NEGATIVE_REWARDS"
    max_entities = 8
    food_cycle_end = 10
    food_value = 150*2
    entity_food_value = 400
    reproduction_threshold = 1200

    BATCH_SIZE = 64_000
    UPDATE_LIMIT = 2

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
