import pygame

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
LIGHT_BLUE = (100, 150, 255)
ORANGE = (255, 165, 0)
PURPLE = (150, 50, 255)
CYAN = (0, 255, 255)



# Fonts (initialized after pygame.init())
font = None
small_font = None
tiny_font = None

def setup_fonts():
    global font, small_font, tiny_font
    import pygame
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    tiny_font = pygame.font.Font(None, 14)

# Export for other modules
__all__ = [
    "BLACK", "WHITE", "BLUE", "RED", "GREEN", "YELLOW", "GRAY", "LIGHT_BLUE", "ORANGE", "PURPLE", "CYAN",
    "font", "small_font", "tiny_font", "Slider", "Checkbox"
]

class Slider:
    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.x + ratio * (self.width - 20)
        handle_rect = pygame.Rect(handle_x, self.y - 5, 20, self.height + 10)
        pygame.draw.rect(screen, YELLOW, handle_rect)
        pygame.draw.rect(screen, WHITE, handle_rect, 2)
        value_text = f"{self.value:.1f}"
        text = small_font.render(value_text, True, WHITE)
        screen.blit(text, (self.x + self.width + 10, self.y - 2))
    def __init__(self, x: int, y: int, width: int, height: int, min_val: float, max_val: float, initial_val: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.rect = pygame.Rect(x, y, width, height)

class Checkbox:
    def __init__(self, x: int, y: int, size: int, label: str, initial_state: bool = False):
        self.x = x
        self.y = y
        self.size = size
        self.label = label
        self.checked = initial_state
        self.rect = pygame.Rect(x, y, size, size)
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.checked = not self.checked
    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        if self.checked:
            pygame.draw.line(screen, GREEN, (self.x + 3, self.y + self.size//2), (self.x + self.size//3, self.y + self.size - 3), 2)
            pygame.draw.line(screen, GREEN, (self.x + self.size//3, self.y + self.size - 3), (self.x + self.size - 3, self.y + 3), 2)
        text = small_font.render(self.label, True, WHITE)
        screen.blit(text, (self.x + self.size + 5, self.y - 2))
