import pygame
from ui_controls import setup_fonts
from simulator import Simulation

pygame.init()
setup_fonts()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()