import pygame
from constants import *


class Menu:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Go Game - Menu')
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)

        self.current_screen = "main"  # "main", "play", "rules"
        self.selected_size = None
        self.play_with_bot = False
        self.bot_vs_bot = False

        # Main menu buttons
        self.main_buttons = [
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2,
                                    WINDOW_SIZE // 2 - BUTTON_HEIGHT - BUTTON_SPACING,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Play',
                'action': 'play'
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2, WINDOW_SIZE // 2 + BUTTON_SPACING,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Rules',
                'action': 'rules'
            }
        ]

        # Play menu buttons
        self.play_buttons = [
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2, WINDOW_SIZE // 3 - BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Play with Human',
                'action': 'human'
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2, WINDOW_SIZE // 3 + BUTTON_SPACING,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Play with Bot',
                'action': 'bot'
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2,
                                    WINDOW_SIZE // 3 + 2 * BUTTON_SPACING + BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Bot vs Bot',
                'action': 'bot_vs_bot'
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2,
                                    WINDOW_SIZE // 3 + 4 * BUTTON_SPACING + 2 * BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Back',
                'action': 'main'
            }
        ]

        # Board size buttons (will be shown after selecting game mode)
        self.size_buttons = [
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2, WINDOW_SIZE // 3 - BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': '9 x 9',
                'size': 9
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2, WINDOW_SIZE // 3 + BUTTON_SPACING,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': '13 x 13',
                'size': 13
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2,
                                    WINDOW_SIZE // 3 + 2 * BUTTON_SPACING + BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': '19 x 19',
                'size': 19
            },
            {
                'rect': pygame.Rect((WINDOW_SIZE - BUTTON_WIDTH) // 2,
                                    WINDOW_SIZE // 3 + 4 * BUTTON_SPACING + 2 * BUTTON_HEIGHT,
                                    BUTTON_WIDTH, BUTTON_HEIGHT),
                'text': 'Back',
                'size': None
            }
        ]

        # Back button for rules screen
        self.back_button = {
            'rect': pygame.Rect(WINDOW_SIZE - BUTTON_WIDTH - 30, WINDOW_SIZE - BUTTON_HEIGHT - 30,
                                BUTTON_WIDTH, BUTTON_HEIGHT),
            'text': 'Back',
            'action': 'main'
        }

        # Go rules text
        self.rules_text = [
            "Basic Rules of Go",
            "",
            "1. Go is played on a grid board (usually 19x19, 13x13, or 9x9)",
            "2. Two players (Black and White) take turns placing stones",
            "3. Black plays first",
            "4. Stones are placed on the intersections of the lines",
            "5. Once placed, stones do not move",
            "6. The goal is to surround more territory than your opponent",
            "",
            "Capturing Stones:",
            "- Stones are captured when they have no liberties (empty adjacent points)",
            "- A group of connected stones is captured when all its liberties are occupied",
            "",
            "End of Game:",
            "- The game ends when both players pass consecutively",
            "- Score is calculated by counting territory and captured stones",
            "- The player with the higher score wins"
        ]

        self.game_mode = None

    def draw(self):
        self.screen.fill(BROWN)

        if self.current_screen == "main":
            self.draw_main_menu()
        elif self.current_screen == "play":
            self.draw_play_menu()
        elif self.current_screen == "rules":
            self.draw_rules()
        elif self.current_screen == "size":
            self.draw_size_selection()

        pygame.display.flip()

    def draw_main_menu(self):
        # Draw title
        title = self.font_large.render('Go Game', True, BLACK)
        title_rect = title.get_rect(centerx=WINDOW_SIZE // 2, y=WINDOW_SIZE // 4)
        self.screen.blit(title, title_rect)

        # Draw buttons
        for button in self.main_buttons:
            pygame.draw.rect(self.screen, LIGHT_BROWN, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

            text = self.font_medium.render(button['text'], True, BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

    def draw_play_menu(self):
        # Draw title
        title = self.font_large.render('Select Game Mode', True, BLACK)
        title_rect = title.get_rect(centerx=WINDOW_SIZE // 2, y=WINDOW_SIZE // 6)
        self.screen.blit(title, title_rect)

        # Draw buttons
        for button in self.play_buttons:
            pygame.draw.rect(self.screen, LIGHT_BROWN, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

            text = self.font_medium.render(button['text'], True, BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

    def draw_size_selection(self):
        # Draw title
        title = self.font_large.render('Select Board Size', True, BLACK)
        title_rect = title.get_rect(centerx=WINDOW_SIZE // 2, y=WINDOW_SIZE // 6)
        self.screen.blit(title, title_rect)

        # Draw buttons
        for button in self.size_buttons:
            pygame.draw.rect(self.screen, LIGHT_BROWN, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

            text = self.font_medium.render(button['text'], True, BLACK)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)

    def draw_rules(self):
        # Draw title
        title = self.font_large.render('Rules of Go', True, BLACK)
        title_rect = title.get_rect(centerx=WINDOW_SIZE // 2, y=50)
        self.screen.blit(title, title_rect)

        # Draw rules text
        y_offset = 120
        for line in self.rules_text:
            if line == "":
                y_offset += 20
                continue

            if "Basic Rules" in line or "Capturing Stones:" in line or "End of Game:" in line:
                text = self.font_medium.render(line, True, BLACK)
            else:
                text = self.font_small.render(line, True, BLACK)

            text_rect = text.get_rect(x=100, y=y_offset)
            self.screen.blit(text, text_rect)
            y_offset += 30

        # Draw back button
        pygame.draw.rect(self.screen, LIGHT_BROWN, self.back_button['rect'])
        pygame.draw.rect(self.screen, BLACK, self.back_button['rect'], 2)

        text = self.font_medium.render(self.back_button['text'], True, BLACK)
        text_rect = text.get_rect(center=self.back_button['rect'].center)
        self.screen.blit(text, text_rect)

    def run(self):
        running = True

        while running:
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, False, False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()

                    if self.current_screen == "main":
                        for button in self.main_buttons:
                            if button['rect'].collidepoint(pos):
                                self.current_screen = button['action']

                    elif self.current_screen == "play":
                        for button in self.play_buttons:
                            if button['rect'].collidepoint(pos):
                                if button['action'] == 'main':
                                    self.current_screen = 'main'
                                else:
                                    self.game_mode = button['action']
                                    self.play_with_bot = (button['action'] == 'bot')
                                    self.bot_vs_bot = (button['action'] == 'bot_vs_bot')
                                    self.current_screen = 'size'

                    elif self.current_screen == "rules":
                        if self.back_button['rect'].collidepoint(pos):
                            self.current_screen = 'main'

                    elif self.current_screen == "size":
                        for button in self.size_buttons:
                            if button['rect'].collidepoint(pos):
                                if button['size'] is None:
                                    self.current_screen = 'play'
                                else:
                                    return button['size'], self.play_with_bot, self.bot_vs_bot

        return None, False, False
