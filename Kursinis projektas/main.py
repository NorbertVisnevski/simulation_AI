import os
import time

import pygame
import sys
import csv

from brain import QLearningControls, HyperParameters, DeepQLearningControls, ClosestQLearningControls
from game_environment import GameEnvironment
from global_data import BLACK
from param_handler import handle_params


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


def main(task):
    # handle_params()
    clock = pygame.time.Clock()
    game = GameEnvironment()

    if not HyperParameters.deep_learning:
        if task == 0:
            carnivoreAI = QLearningControls(HyperParameters.AI_DIRECTORY)
            herbivoreAI = QLearningControls(HyperParameters.AI_DIRECTORY)
        else:
            carnivoreAI = ClosestQLearningControls(HyperParameters.AI_DIRECTORY)
            herbivoreAI = ClosestQLearningControls(HyperParameters.AI_DIRECTORY)
    else:
        carnivoreAI = DeepQLearningControls(HyperParameters.AI_DIRECTORY)
        herbivoreAI = DeepQLearningControls(HyperParameters.AI_DIRECTORY)

    os.makedirs(f"{os.getcwd()}/{HyperParameters.AI_DIRECTORY}", exist_ok=True)

    try:
        carnivoreAI.load("carnivore" + str(HyperParameters._episode))
        herbivoreAI.load("herbivore" + str(HyperParameters._episode))
        print("loaded agents")
    except:
        print("failed to load agents")

    game.carnivoreAI = carnivoreAI
    game.herbivoreAI = herbivoreAI

    pygame.init()
    font = pygame.font.SysFont('didot.ttc', 24)
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.display.set_caption("Agents")
    camera = Camera()

    game.reset_simulation()
    start_time = time.time()
    frametimes = []
    draw = False
    render_over = False

    stats=[]
    try:
        with open(f"{HyperParameters.AI_DIRECTORY}/stats.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                stats.append(row)
            stats.pop(0)
    except:
        pass
    header = ["episode",
              "iterations",
              "epsilon",
              "time",
              "car_avg",
              "car_min",
              "car_max",
              "car_cumulative",
              "car_reproductions",
              "car_table_size",
              "her_avg",
              "her_min",
              "her_max",
              "her_cumulative",
              "her_reproductions",
              "her_table_size",
              ]
    while True:
        frame_start_time = time.time()
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # print(sum(frametimes) / len(frametimes))
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        keys = pygame.key.get_pressed()

        if keys[pygame.K_TAB]:
            draw = not draw

        if game.is_simulation_dead() or HyperParameters.reached_max_iterations():

            render_over = True
            if HyperParameters.learning:
                end_time = time.time()

                episode_stats = []

                episode_stats.append(HyperParameters.episode)
                episode_stats.append(HyperParameters.iteration)
                episode_stats.append(HyperParameters.epsilon)
                episode_stats.append(end_time - start_time)

                avg, min, max, cumulative, reproductions = carnivoreAI.calculate_stats()

                episode_stats.append(avg)
                episode_stats.append(min)
                episode_stats.append(max)
                episode_stats.append(cumulative)
                episode_stats.append(reproductions)
                episode_stats.append(len(carnivoreAI.Q_Table.keys()))

                avg, min, max, cumulative, reproductions = herbivoreAI.calculate_stats()

                episode_stats.append(avg)
                episode_stats.append(min)
                episode_stats.append(max)
                episode_stats.append(cumulative)
                episode_stats.append(reproductions)
                episode_stats.append(len(herbivoreAI.Q_Table.keys()))

                stats.append(episode_stats)

                start_time = time.time()

                # frame_end_time = time.time()
                # frametimes.append(frame_end_time - frame_start_time)

                HyperParameters.decay_epsilon()
                # if HyperParameters.reached_max_iterations():
                with open(f"{HyperParameters.AI_DIRECTORY}/stats.csv", 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(stats)
                # carnivoreAI.save("carnivore")
                # herbivoreAI.save("herbivore")
                # HyperParameters.epsilon = 1
                game.reset_simulation()
                render_over = False
                HyperParameters.episode += 1
                # HyperParameters.iteration = 0
                if HyperParameters.learning:
                    carnivoreAI.learn()
                    herbivoreAI.learn()
                    # HyperParameters.iteration += 1
                if HyperParameters.episode == 10001:
                    return
            if keys[pygame.K_SPACE]:
                game.reset_simulation()
                render_over = False
        else:
            camera.update()
            game.update()

        if draw or not HyperParameters.learning:
            game.draw(screen, camera)
        screen.blit(font.render(f"Carnivores: {game.carnivores}", True, BLACK), (0, 0))
        screen.blit(font.render(f"Herbivores: {game.herbivores}", True, BLACK), (0, 36))
        screen.blit(font.render(f"Episode: {HyperParameters.episode}", True, BLACK), (0, 72))
        screen.blit(font.render(f"Epsilon: {round(HyperParameters.epsilon,5)}", True, BLACK), (0, 108))

        if render_over:
            screen.blit(font.render(f"Simulation over!", True, BLACK), (screen.get_width() / 2 - 100, 250))
            screen.blit(font.render(f"All {'carnivores' if game.carnivores == 0 else 'herbivores'} have been eliminated", True, BLACK), (screen.get_width() / 2 - 170, 300))
            screen.blit(font.render(f"Press space to restart", True, BLACK), (screen.get_width() / 2 - 120, 350))

        pygame.display.update()
        # if HyperParameters.learning:
        #     carnivoreAI.learn()
        #     herbivoreAI.learn()
        #     HyperParameters.iteration += 1
        # frame_end_time = time.time()
        # frametimes.append(frame_end_time - frame_start_time)
        if not HyperParameters.learning:
            clock.tick(60)


if __name__ == '__main__':
    for i in range(3):
        HyperParameters.episode = 1
        HyperParameters._episode = 0
        HyperParameters.epoch = 0
        HyperParameters.iteration = 0
        HyperParameters.epsilon = 1
        HyperParameters.AI_DIRECTORY = "DEFAULT_AVERAGE2/" + str(i)
        main(0)
    for i in range(3):
        HyperParameters.episode = 1
        HyperParameters._episode = 0
        HyperParameters.epoch = 0
        HyperParameters.iteration = 0
        HyperParameters.epsilon = 1
        HyperParameters.AI_DIRECTORY = "CLOSEST_AVERAGE2/" + str(i)
        main(1)
