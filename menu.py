# menu.py
import pygame
from constants import *


class Menu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Go-Game - Chose size')
        self.font = pygame.font.Font(None, 48)
        self.selected_size = None

        button_height = 60
        button_width = 200
        spacing = 30
        start_y = WINDOW_SIZE // 2 - (3 * button_height + 2 * spacing) // 2

        self.buttons = [
            {
                'rect': pygame.Rect((WINDOW_SIZE - button_width) // 2, start_y, button_width, button_height),
                'text': '9 x 9',
                'size': 9
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - button_width) // 2, start_y + button_height + spacing, button_width,
                                    button_height),
                'text': '13 x 13',
                'size': 13
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - button_width) // 2, start_y + 2 * (button_height + spacing),
                                    button_width, button_height),
                'text': '19 x 19',
                'size': 19
            }
        ]

    def draw(self):
        self.screen.fill(BROWN)

        title = self.font.render('Chose size', True, BLACK)
        title_rect = title.get_rect(centerx=WINDOW_SIZE // 2, y=100)
        self.screen.blit(title, title_rect)

        # Vẽ các nút
        for button in self.buttons:
            # Vẽ nền nút
            pygame.draw.rect(self.screen, WHITE, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

            # Vẽ text
            text = self.font.render(button['text'], True, BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in self.buttons:
                if button['rect'].collidepoint(mouse_pos):
                    self.selected_size = button['size']
                    return True
        return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if self.handle_event(event):
                    return self.selected_size

            self.draw()
        return None

