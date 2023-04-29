import math

import game_environment
from brain import *
from game_environment import *
from entity import *
from global_data import GREEN1, BLUE, RED, GREEN


class Agent(Entity):
    average_speed = 10
    jump = 100
    food_sensor_length = 100
    touch_sensor_length = 60
    food_sensor_spread = 0.7
    forward_vec_size = 25
    radius = 25

    def __init__(self, game_env, point=pygame.math.Vector2(), rotation=math.pi, brain=PlayerControls()):
        self.brain = brain
        self.game_environment = game_env
        self.coordinates: pygame.math.Vector2 = point
        self.angle = rotation
        self.energy = 900
        self.food_value = HyperParameters.entity_food_value
        self.reproduction_threshold = HyperParameters.reproduction_threshold
        self.max_energy = self.reproduction_threshold
        self.state = None
        self.action = 0
        self.reward = 0
        self.rotation = math.pi * 2
        self.food_sensors = None
        self.touch_sensors = None
        self.update_body()
        self.state = self.get_state()

    def update_body(self):
        self.food_sensors = [
            (self.coordinates.x - math.sin(self.angle + self.food_sensor_spread) * self.food_sensor_length,
             self.coordinates.y + math.cos(self.angle + self.food_sensor_spread) * self.food_sensor_length),
            (self.coordinates.x - math.sin(self.angle - self.food_sensor_spread) * self.food_sensor_length,
             self.coordinates.y + math.cos(self.angle - self.food_sensor_spread) * self.food_sensor_length)
        ]
        self.touch_sensors = [
            (self.coordinates.x - math.sin(self.angle) * self.touch_sensor_length,
             self.coordinates.y + math.cos(self.angle) * self.touch_sensor_length),
            (self.coordinates.x - math.sin(self.angle + math.pi / 2) * self.touch_sensor_length,
             self.coordinates.y + math.cos(self.angle + math.pi / 2) * self.touch_sensor_length),
            (self.coordinates.x - math.sin(self.angle - math.pi / 2) * self.touch_sensor_length,
             self.coordinates.y + math.cos(self.angle - math.pi / 2) * self.touch_sensor_length),
            (self.coordinates.x - math.sin(self.angle + math.pi) * self.touch_sensor_length,
             self.coordinates.y + math.cos(self.angle + math.pi) * self.touch_sensor_length),
        ]

    def update(self):
        self.state = self.get_state()
        self.action: int = self.brain.get_action(self.state)
        if self.action == 0:
            new_coords = pygame.math.Vector2(self.coordinates.x - math.sin(self.angle) * self.average_speed,
                                             self.coordinates.y + math.cos(self.angle) * self.average_speed)
            if self.game_environment.map.is_in_bounds(new_coords):
                self.coordinates = new_coords
            else:
                self.rotate(math.pi)
            self.energy -= 1
        elif self.action == 1:
            self.rotate(-0.1)
            self.energy -= 1
        elif self.action == 2:
            self.rotate(0.1)
            self.energy -= 1
        elif self.action == 3:
            if self.jump > self.energy:
                new_coords = pygame.math.Vector2(self.coordinates.x - math.sin(self.angle) * self.jump,
                                                 self.coordinates.y + math.cos(self.angle) * self.jump)
                if self.game_environment.map.is_in_bounds(new_coords):
                    self.coordinates = new_coords
                self.energy -= self.jump/10
            else:
                new_coords = pygame.math.Vector2(self.coordinates.x - math.sin(self.angle) * self.average_speed,
                                                 self.coordinates.y + math.cos(self.angle) * self.average_speed)
                if self.game_environment.map.is_in_bounds(new_coords):
                    self.coordinates = new_coords
                else:
                    self.rotate(math.pi)
                self.energy -= 1

        if self.energy <= 0:
            self.reward = -100
            self.game_environment.remove(self)
            return

        self.update_body()

        if self.tag == "herbivore":
            for i in range(len(self.game_environment.food) - 1, -1, -1):
                distance = pygame.math.Vector2.distance_to(self.coordinates, self.game_environment.food[i].coordinates)
                if distance <= self.radius + self.game_environment.food[i].radius:
                    self.energy += self.game_environment.food[i].food_value
                    del self.game_environment.food[i]
                    self.reward = 10

        if self.tag == "carnivore":
            for entity in self.game_environment.get_entities():
                if entity.tag == "herbivore":
                    distance = pygame.math.Vector2.distance_to(self.coordinates, entity.coordinates)
                    if distance <= self.radius + entity.radius:
                        entity.reward = -100
                        self.energy += entity.food_value
                        self.game_environment.remove(entity)
                        self.reward = 20

        if self.energy > self.reproduction_threshold:
            child = Agent(self.game_environment, self.coordinates.copy(), random.randrange(6), self.brain)
            child.tag = self.tag
            child.energy = self.energy/2
            child.state = child.get_state()
            self.game_environment.add(child)
            self.energy = self.energy/2
            self.reward = 100

    def draw(self, screen, camera):
        pygame.draw.line(screen, GREEN1, self.coordinates + camera.offset, self.food_sensors[0] + camera.offset, 3)
        pygame.draw.line(screen, GREEN1, self.coordinates + camera.offset, self.food_sensors[1] + camera.offset, 3)

        for touch_sensor in self.touch_sensors:
            pygame.draw.line(screen, BLUE, self.coordinates + camera.offset,
                             touch_sensor + camera.offset, 1)

        pygame.draw.circle(screen, RED if self.tag == "carnivore" else GREEN, self.coordinates + camera.offset, self.radius)

    def rotate(self, rads):
        self.angle += rads
        if self.angle > self.rotation:
            self.angle -= self.rotation
        elif self.angle < -0:
            self.angle += self.rotation

    def get_state(self):
        food = abs(self.energy / self.max_energy)
        environment_state = [food if food <= 1 else 1.0]
        if self.tag == "herbivore":
            for sensor in self.food_sensors:
                sensor_values = [0.0]
                for food in self.game_environment.food:
                    try:
                        distance = pygame.math.Vector2.distance_to(pygame.math.Vector2(sensor), food.coordinates)
                        sensor_values.append(100 / distance)
                    except:
                        sensor_values.append(1.0)
                max_val = max(sensor_values)
                environment_state.append(1.0 if max_val > 1 else max_val)
        elif self.tag == "carnivore":
            for sensor in self.food_sensors:
                sensor_values = [0.0]
                for entity in self.game_environment.entities:
                    if entity != self and entity.tag == "herbivore":
                        try:
                            distance = pygame.math.Vector2.distance_to(pygame.math.Vector2(sensor), entity.coordinates)
                            sensor_values.append(100 / distance)
                        except:
                            sensor_values.append(1.0)
                max_val = max(sensor_values)
                environment_state.append(1.0 if max_val > 1 else max_val)

        entities = self.game_environment.get_entities()
        # entities.extend(self.game_environment.food)
        for sensor in self.touch_sensors:
            added = False
            # if not self.game_environment.map.is_in_bounds(pygame.math.Vector2(sensor)):
            #     environment_state.append(1.14)
            #     continue
            for entity in entities:
                if entity != self:
                    distance = pygame.math.Vector2.distance_to(pygame.math.Vector2(sensor), entity.coordinates)
                    if distance <= entity.radius * 1.5:
                        environment_state.append(1.0)
                        added = True
                        break
            if not added:
                environment_state.append(0.0)

        return environment_state

    def store_experience(self):
        next_state = self.get_state()
        # if self.reward == 14:
        #     if self.state[1] + self.state[2] < next_state[1] + next_state[2]:
        #         self.reward = 1
        #     else:
        #         self.reward = -1
        if self.reward == 0:
            self.reward = (next_state[1] + next_state[2]) - (self.state[1] + self.state[2])
            # if self.reward < 14:
            #     self.reward = 14
        self.brain.store(self.state, self.action, self.reward, next_state)
        self.reward = 0


class Herbivore(Agent):
    tag = "herbivore"

    def __init__(self, game_env, point=pygame.math.Vector2(), rotation=math.pi, brain=PlayerControls()):
        super().__init__(game_env, point, rotation, brain)


class Carnivore(Agent):
    tag = "carnivore"

    def __init__(self, game_env, point=pygame.math.Vector2(), rotation=math.pi, brain=PlayerControls()):
        super().__init__(game_env, point, rotation, brain)
