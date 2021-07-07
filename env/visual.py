import sys
import numpy as np
import pygame

# Screen properties
SCREEN_WIDTH = 500; SCREEN_HEIGHT = 500

# Grid properties
GRID_SIZE = 50
GRID_WIDTH = SCREEN_WIDTH / GRID_SIZE; GRID_HEIGHT = SCREEN_HEIGHT / GRID_SIZE

TILE_COLOR_ONE = (247, 247, 247)
TILE_COLOR_TWO = (235, 235, 235)

# Temperature properties
TEMP_NUM_COLOR = (0, 0, 0)

# Pygame font
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 27)

def convert_pos(x, y):
    return (x * GRID_SIZE + (GRID_SIZE / 2), y * GRID_SIZE + (GRID_SIZE / 2)) 

class Visual:
    def __init__(self, data: np.ndarray):
        self.init_render = False
        self.data = data
    
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
        pygame.draw.rect(surface, (93, 216, 228), rect, 1) # border rect

        temp = FONT.render(str(value), 1, TEMP_NUM_COLOR)
        text_rect = temp.get_rect(center=(position[0], position[1]))
        surface.blit(temp, text_rect)
    
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

        self.screen.blit(self.surface, (0,0))

        pygame.display.update()
