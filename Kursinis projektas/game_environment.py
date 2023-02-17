import os
import random
import time

import pygame
from agent import Herbivore, Carnivore
from brain import QLearningControls, ClosestQLearningControls
from entity import *
from global_data import FOOD_COLOR, HyperParameters


class Map(Entity):
    tag = "map"
    x = int(HyperParameters.MAP_X)
    y = int(HyperParameters.MAP_Y)
    thickness = 5

    def __init__(self):
        self.rect = pygame.rect.Rect(0, 0, self.x, self.y)

    def draw(self, screen, camera):
        pygame.draw.rect(screen, (255, 0, 0), pygame.rect.Rect(camera.offset, (self.x, self.y)), self.thickness)

    def is_in_bounds(self, location):
        return self.rect.collidepoint(location)


class Food(Entity):
    tag = "food"
    radius = 25
    food_value = HyperParameters.food_value

    def __init__(self, location):
        self.coordinates = location

    def draw(self, screen, camera):
        pygame.draw.circle(screen, FOOD_COLOR, self.coordinates + camera.offset, self.radius)


class GameEnvironment(Entity):

    def __init__(self):
        self.carnivoreAI = None
        self.herbivoreAI = None
        self.entities = list()
        self.map = Map()
        self.food = []
        self.max_entities = HyperParameters.max_entities
        self.food_count = self.max_entities * 8
        self.food_cycle = 0
        self.food_cycle_end = HyperParameters.food_cycle_end
        self.dead = []
        self.herbivores = 0
        self.carnivores = 0
        self.herbivoreAIs = []
        self.carnivoreAIs = []

    def place_food(self, force=False):
        if self.food_cycle >= self.food_cycle_end or force:
            if len(self.food) < self.food_count or force:
                self.food.append(Food(pygame.math.Vector2(random.randrange(self.map.x), random.randrange(self.map.y))))
            self.food_cycle = 0
        else:
            self.food_cycle += 1

    def reset_simulation(self):
        self.entities.clear()
        self.food.clear()
        self.carnivores = 0
        self.herbivores = 0
        for i in range(self.max_entities):
            herbivore = Herbivore(self, pygame.math.Vector2(random.randrange(self.map.x), random.randrange(self.map.y)), random.randrange(6), self.herbivoreAIs[i])
            self.add(herbivore)
            carnivore = Carnivore(self, pygame.math.Vector2(random.randrange(self.map.x), random.randrange(self.map.y)), random.randrange(6), self.carnivoreAIs[i])
            self.add(carnivore)
            self.place_food(True)

    def update(self):
        self.place_food()
        for entity in self.entities:
            entity.update()
        self.store_experience()
        self.dead.clear()

    def is_simulation_dead(self):
        return self.herbivores == 0 or self.carnivores == 0

    def store_experience(self):
        for entity in self.entities:
            entity.store_experience()
        for entity in self.dead:
            entity.store_experience()

    def draw(self, screen, camera):
        self.map.draw(screen, camera)
        for entity in self.entities:
            entity.draw(screen, camera)
        for food in self.food:
            food.draw(screen, camera)

    def remove(self, entity):
        if entity.tag == "herbivore":
            self.herbivores -= 1
        elif entity.tag == "carnivore":
            self.carnivores -= 1
        self.entities.remove(entity)
        self.dead.append(entity)

    def add(self, entity):
        if entity.tag == "herbivore":
            self.herbivores += 1
            self.herbivoreAIs.append(entity.brain.copy())
        elif entity.tag == "carnivore":
            self.carnivores += 1
            self.carnivoreAIs.append(entity.brain.copy())
        self.entities.append(entity)

    def get_entities(self):
        return list(self.entities)

    def generate_genomes(self, carnivoreAI, herbivoreAI):
        start_time = time.time()
        self.herbivoreAIs.clear()
        self.carnivoreAIs.clear()
        for i in range(self.max_entities):
            self.herbivoreAIs.append(herbivoreAI.copy())
            self.carnivoreAIs.append(carnivoreAI.copy())
        end_time = time.time()
        print(end_time - start_time, 'time')

    def get_best_AI(self):
        # print(len(self.carnivoreAIs), len(self.herbivoreAIs))
        # start_time = time.time()
        carnivoreAI = self.carnivoreAIs[0]
        for i in range(1, len(self.carnivoreAIs)):
            b_avg, b_min, b_max, b_cumulative, b_reproductions = carnivoreAI.calculate_stats()
            t_avg, t_min, t_max, t_cumulative, t_reproductions = self.carnivoreAIs[i].calculate_stats()
            if t_cumulative > b_cumulative:
                print(t_cumulative, b_cumulative)
                carnivoreAI = self.carnivoreAIs[i]

        herbivoreAI = self.herbivoreAIs[0]
        for i in range(1, len(self.herbivoreAIs)):
            b_avg, b_min, b_max, b_cumulative, b_reproductions = herbivoreAI.calculate_stats()
            t_avg, t_min, t_max, t_cumulative, t_reproductions = self.herbivoreAIs[i].calculate_stats()
            if t_cumulative > b_cumulative:
                herbivoreAI = self.herbivoreAIs[i]

        # end_time = time.time()
        # print(end_time - start_time, 'time')

        return carnivoreAI, herbivoreAI