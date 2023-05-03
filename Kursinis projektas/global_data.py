class HyperParameters:
    learning_rate = 0.01
    inverse_alpha = 1 - learning_rate
    discount = 0.3
    epsilon = 1
    epsilon_decay = 0.95
    epoch = 0
    observation_size = 3
    action_size = 4
    replay_memory_size = 2000

    _episode = 0
    episode = _episode + 1
    learning = True
    deep_learning = False
    AI_DIRECTORY = "MULTIPLE_GENOMES/test"
    max_entities = 6
    food_cycle_end = 10
    food_value = 300
    entity_food_value = 400
    reproduction_threshold = 1200

    episode_iteration_count = 100_000
    iteration = 0

    BATCH_SIZE = 64_000
    UPDATE_LIMIT = 2

    MAP_X = 4_000
    MAP_Y = 2_000

    @staticmethod
    def decay_epsilon():
        HyperParameters.epsilon *= HyperParameters.epsilon_decay
        # if HyperParameters.epsilon < 0.01:
        #     HyperParameters.epsilon = 0

    @staticmethod
    def reached_max_iterations():
        return (HyperParameters.episode_iteration_count == HyperParameters.iteration) and HyperParameters.learning


BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
GREEN1 = (0, 150, 0)
GREEN2 = (0, 150, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
FOOD_COLOR = (10, 200, 150)
