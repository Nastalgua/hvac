import sys
import numpy as np
import pygame
from pygame import surface
from pygame import color

# Screen properties
SCREEN_WIDTH = 500; SCREEN_HEIGHT = 500

# Grid properties
GRID_SIZE = 50
GRID_WIDTH = SCREEN_WIDTH / GRID_SIZE; GRID_HEIGHT = SCREEN_HEIGHT / GRID_SIZE

# Colors
TILE_COLOR_ONE = (247, 247, 247)
TILE_COLOR_TWO = (235, 235, 235)

AC_COLOR = (93, 216, 228)
TARGET_COLOR = (91, 128, 207)

TEMP_NUM_COLOR = (0, 0, 0)

# Pygame font
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 27)

def convert_pos(x, y):
    return (x * GRID_SIZE + (GRID_SIZE / 2), y * GRID_SIZE + (GRID_SIZE / 2)) 

class Visual:
    def __init__(self, data: np.ndarray, acs: np.ndarray, targets: np.ndarray):
        self.init_render = False
        self.data = data
        self.acs = acs
        self.targets = targets
    
    def updateData(self, data: np.ndarray):
        self.data = data

    def draw_grid(self, surface):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                if (x + y) % 2 == 0:
                    rect = pygame.Rect((x * GRID_SIZE, y * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
                    pygame.draw.rect(surface, TILE_COLOR_ONE, rect)
                else:
                    rect = pygame.Rect((x * GRID_SIZE, y * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
                    pygame.draw.rect(surface, TILE_COLOR_TWO, rect)

    def draw_tile(self, position, value, surface):
        value = round(value, 0)
        rect = pygame.Rect((position[0] - (GRID_SIZE / 2), position[1] - (GRID_SIZE / 2)), (GRID_SIZE, GRID_SIZE))

        pygame.draw.rect(surface, (222, min(max(222 - value, 0), 255), min(max(222 - value, 0), 255)), rect)

        temp = FONT.render(str(value), 1, TEMP_NUM_COLOR)
        text_rect = temp.get_rect(center=(position[0], position[1]))
        surface.blit(temp, text_rect)
    
    def draw_special(self, position, color, surface):
        rect = pygame.Rect(
            (position[0] - (GRID_SIZE / 2), position[1] - (GRID_SIZE / 2)), 
            (GRID_SIZE, GRID_SIZE)
        )

        pygame.draw.rect(surface, color, rect, 1) # border rect

    def render(self):
        if not self.init_render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

            self.surface = pygame.Surface(self.screen.get_size())
            self.surface = self.surface.convert()

            self.init_render = True # prevent this code block from running again

        # key handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: # key press test
                    pass

        self.draw_grid(surface=self.surface)

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.draw_tile(convert_pos(i, j), self.data[i][j], surface=self.surface)

        for target in self.targets:
            target_pos = target.position
            self.draw_special(convert_pos(target_pos[0], target_pos[1]), color=TARGET_COLOR, surface=self.surface)
        
        for ac in self.acs:
            ac_pos = ac.position
            self.draw_special(convert_pos(ac_pos[0], ac_pos[1]), color=AC_COLOR, surface=self.surface)

        self.screen.blit(self.surface, (0,0))

        pygame.display.update()
