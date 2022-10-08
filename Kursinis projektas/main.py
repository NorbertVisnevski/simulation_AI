import os
import time

import pygame
import sys
import csv

from brain import QLearningControls, HyperParameters, DeepQLearningControls
from game_environment import Map, GameEnvironment
from agent import Agent
from global_data import BLACK


class Camera:
    def __init__(self):
        self.offset = pygame.Vector2()

    def update(self):
        keys = pygame.key.get_pressed()
        speed = 10
        if keys[pygame.K_LSHIFT]:
            speed = 50
        if keys[pygame.K_w]:
            self.offset.update(pygame.Vector2(self.offset.x, self.offset.y + speed))
        if keys[pygame.K_a]:
            self.offset.update(pygame.Vector2(self.offset.x + speed, self.offset.y))
        if keys[pygame.K_s]:
            self.offset.update(pygame.Vector2(self.offset.x, self.offset.y - speed))
        if keys[pygame.K_d]:
            self.offset.update(pygame.Vector2(self.offset.x - speed, self.offset.y))


def main():
    clock = pygame.time.Clock()
    learning = True
    HyperParameters.epsilon = 1
    HyperParameters.episode = 1
    game = GameEnvironment()

    AI_DIRECTORY = "Q_LEARNING_EPISODE"
    carnivoreAI = QLearningControls(AI_DIRECTORY)
    herbivoreAI = QLearningControls(AI_DIRECTORY)
    # AI_DIRECTORY = "DEEP_Q_LEARNING_EPISODE_SMALL_NET"
    # carnivoreAI = DeepQLearningControls(AI_DIRECTORY)
    # herbivoreAI = DeepQLearningControls(AI_DIRECTORY)
    os.makedirs(f"{os.getcwd()}/{AI_DIRECTORY}", exist_ok=True)

    # carnivoreAI.load("carnivore323594")
    # herbivoreAI.load("herbivore323594")

    game.carnivoreAI = carnivoreAI
    game.herbivoreAI = herbivoreAI

    pygame.init()
    font = pygame.font.SysFont('didot.ttc', 24)
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.display.set_caption("Agents")
    camera = Camera()

    game.reset_simulation()
    start_time = time.time()
    draw = False

    stats=[]

    # with open(f"{AI_DIRECTORY}/stats.csv") as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         stats.append(row)
    #     stats.pop(0)
    header = ["episode",
              "epsilon",
              "time",
              "car_avg",
              "car_min",
              "car_max",
              "her_avg",
              "her_min",
              "her_max"]

    while True:

        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_TAB]:
            draw = not draw

        if game.is_simulation_dead():
            if learning:
                end_time = time.time()

                episode_stats = []

                episode_stats.append(HyperParameters.episode)
                episode_stats.append(HyperParameters.epsilon)
                episode_stats.append(end_time - start_time)

                avg, min, max = carnivoreAI.calculate_stats()

                episode_stats.append(avg)
                episode_stats.append(min)
                episode_stats.append(max)

                avg, min, max = herbivoreAI.calculate_stats()

                episode_stats.append(avg)
                episode_stats.append(min)
                episode_stats.append(max)

                stats.append(episode_stats)

                carnivoreAI.learn()
                herbivoreAI.learn()

                HyperParameters.decay_epsilon()
                if HyperParameters.epsilon == 0:
                    with open(f"{AI_DIRECTORY}/stats.csv", 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(stats)
                    carnivoreAI.save("carnivore")
                    herbivoreAI.save("herbivore")
                    HyperParameters.epsilon = 1
            game.reset_simulation()
            HyperParameters.episode += 1
            start_time = time.time()

        camera.update()
        game.update()
        # if learning:
        #     carnivoreAI.learn()
        #     herbivoreAI.learn()
        # if
        # carnivoreAI.learn()
        # herbivoreAI.learn()

        if draw:
            game.draw(screen, camera)
        screen.blit(font.render(f"Carnivores: {game.carnivores}", True, BLACK), (0, 0))
        screen.blit(font.render(f"Herbivores: {game.herbivores}", True, BLACK), (0, 36))
        screen.blit(font.render(f"Episode: {HyperParameters.episode}", True, BLACK), (0, 72))
        screen.blit(font.render(f"Epsilon: {round(HyperParameters.epsilon,5)}", True, BLACK), (0, 108))

        pygame.display.update()

        # clock.tick(60)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
