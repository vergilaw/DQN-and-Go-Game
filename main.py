from game import GoGame
from menu import Menu
from bot import MCTS
import pygame
from constants import CELL_SIZE, WINDOW_SIZE
import time


def main():
    pygame.init()

    menu = Menu()
    board_size, play_with_bot = menu.run()

    if board_size is None:
        pygame.quit()
        return

    game = GoGame()
    game.change_board_size(board_size)
    running = True

    bot = None
    if play_with_bot:
        bot = MCTS(time_limit=5.0)  # thinking time

    bot_pass_message = None
    bot_pass_time = 0
    message_duration = 2.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Pass
                    game.make_move('pass')
                    if play_with_bot and game.current_player == 'white' and not game.show_score:
                        pygame.display.flip()
                        bot_move = bot.get_move(game)
                        if bot_move:
                            if bot_move == 'pass':

                                bot_pass_message = "Bot PASSED!"
                                bot_pass_time = time.time()
                            game.make_move(bot_move)
                elif event.key == pygame.K_r:  # Reset game
                    game.reset_game()
                    bot_pass_message = None
                elif event.key == pygame.K_m:  # Back to menu
                    board_size, play_with_bot = menu.run()
                    if board_size is not None:
                        game.change_board_size(board_size)
                        if play_with_bot:
                            bot = MCTS(time_limit=2.0)
                    bot_pass_message = None

            elif event.type == pygame.MOUSEBUTTONDOWN and not game.show_score:
                # player click
                if not play_with_bot or game.current_player == 'black':
                    x, y = event.pos
                    grid_x = round((x - game.board.margin) / CELL_SIZE)
                    grid_y = round((y - game.board.margin) / CELL_SIZE)

                    if 0 <= grid_x < game.board.size and 0 <= grid_y < game.board.size:
                        if game.make_move((grid_x, grid_y)):
                            # bot
                            if play_with_bot and game.current_player == 'white' and not game.show_score:
                                game.draw()
                                pygame.display.flip()

                                font = pygame.font.Font(None, 36)
                                thinking_text = font.render("Bot thinking...", True, (255, 0, 0))
                                game.screen.blit(thinking_text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE - 50))
                                pygame.display.flip()

                                bot_move = bot.get_move(game)
                                if bot_move:
                                    if bot_move == 'pass':
                                        #Pass
                                        bot_pass_message = "Bot PASSED!"
                                        bot_pass_time = time.time()
                                    game.make_move(bot_move)

        game.draw()

        # Hiển thị thông báo bot pass nếu có
        if bot_pass_message and time.time() - bot_pass_time < message_duration:
            font = pygame.font.Font(None, 48)
            pass_text = font.render(bot_pass_message, True, (255, 0, 0))
            text_rect = pass_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 100))
            pygame.draw.rect(game.screen, (255, 255, 200),
                             (text_rect.x - 10, text_rect.y - 10,
                              text_rect.width + 20, text_rect.height + 20))
            pygame.draw.rect(game.screen, (0, 0, 0),
                             (text_rect.x - 10, text_rect.y - 10,
                              text_rect.width + 20, text_rect.height + 20), 2)
            game.screen.blit(pass_text, text_rect)
        else:
            bot_pass_message = None

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
