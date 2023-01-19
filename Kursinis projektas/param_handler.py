from brain import HyperParameters

def handle_params():
    try:
        value = input(f"learning_rate={HyperParameters.learning_rate} ")
        if value != '':
            HyperParameters.learning_rate = float(value)

        value = input(f"discount={HyperParameters.discount} ")
        if value != '':
            HyperParameters.discount = float(value)

        value = input(f"epsilon={HyperParameters.epsilon} ")
        if value != '':
            HyperParameters.epsilon = float(value)

        value = input(f"epsilon_decay={HyperParameters.epsilon_decay} ")
        if value != '':
            HyperParameters.epsilon_decay = float(value)

        value = input(f"_episode={HyperParameters._episode} ")
        if value != '':
            HyperParameters._episode = int(value)

        value = input(f"learning={HyperParameters.learning} ")
        if value != '':
            HyperParameters.learning = bool(value)

        value = input(f"deep_learning={HyperParameters.deep_learning} ")
        if value != '':
            HyperParameters.deep_learning = bool(value)

        value = input(f"AI_DIRECTORY={HyperParameters.AI_DIRECTORY} ")
        if value != '':
            HyperParameters.AI_DIRECTORY = value

        value = input(f"max_entities={HyperParameters.max_entities} ")
        if value != '':
            HyperParameters.max_entities = int(value)

        value = input(f"food_cycle_end={HyperParameters.food_cycle_end} ")
        if value != '':
            HyperParameters.food_cycle_end = int(value)

        value = input(f"food_value={HyperParameters.food_value} ")
        if value != '':
            HyperParameters.food_value = int(value)

        value = input(f"BATCH_SIZE={HyperParameters.BATCH_SIZE} ")
        if value != '':
            HyperParameters.BATCH_SIZE = int(value)

        value = input(f"UPDATE_LIMIT={HyperParameters.UPDATE_LIMIT} ")
        if value != '':
            HyperParameters.UPDATE_LIMIT = int(value)
    except:
        print("parsing error - reverting to default values")


