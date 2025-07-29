import pygame
from ui_controls import setup_fonts
from simulator_player import Simulation

# Initialize Pygame and custom fonts
pygame.init()
setup_fonts()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()