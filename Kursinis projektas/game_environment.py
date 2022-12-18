import random
import pygame
from agent import Herbivore, Carnivore
from entity import *
from global_data import FOOD_COLOR


class Map(Entity):
    tag = "map"
    x = int(2500)
    y = int(1400)
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
    food_value = 150

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
        self.max_entities = 6
        self.food_count = self.max_entities * 10
        self.food_cycle = 0
        self.food_cycle_end = 10
        self.dead = []
        self.herbivores = 0
        self.carnivores = 0

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
            herbivore = Herbivore(self, pygame.math.Vector2(random.randrange(self.map.x), random.randrange(self.map.y)), random.randrange(6), self.herbivoreAI)
            carnivore = Carnivore(self, pygame.math.Vector2(random.randrange(self.map.x), random.randrange(self.map.y)), random.randrange(6), self.carnivoreAI)
            self.add(herbivore)
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
        elif entity.tag == "carnivore":
            self.carnivores += 1
        self.entities.append(entity)

    def get_entities(self):
        return list(self.entities)
