import numpy as np
import pygame
import threading
import os
import sys
import math
from math import e
from concurrent import futures
from functii.teste2 import generate_matrix
import ctypes
thread_pool_executor = futures.ThreadPoolExecutor(max_workers=2)
ctypes.windll.user32.SetProcessDPIAware()

class Program:
    pygame.init()
    clock = pygame.time.Clock()
    infoObject = pygame.display.Info()
    screen=pygame.display.set_mode((800, 800))
    number=0
    number2=0
    run=True
    run1=False
    font = pygame.font.SysFont('consolas', bold=True, size=16)

    def __init__(self):
        thread = threading.Thread(target=self.run_algorithm)
        thread.start()

    def run_algorithm(self):
            if self.run1 is True:
                generate_matrix()


    def display_text(self,text,x,y):
        text_display = self.font.render(text, True,(255, 255, 255))
        self.screen.blit(text_display,(x,y))

    def show_canvas(self):
        while self.run:
            self.screen.fill((174, 189, 214))
            if self.number<500:
                self.number+=1
            else:
                self.number=0
                self.number2+=1
            self.display_text(str(self.number2),100,100)
            pygame.draw.rect(self.screen, (0, 0, 255),
                             pygame.Rect(300, 300, 64, 64))
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.run=False
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if mx>300 and mx<364 and my>300 and my<364:
                            self.run1=True
            pygame.display.update()

program=Program()
program.show_canvas()