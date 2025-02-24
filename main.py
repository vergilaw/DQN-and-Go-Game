from game import GoGame
from menu import Menu
import pygame
from constants import CELL_SIZE


def main():
    pygame.init()


    menu = Menu()
    board_size = menu.run()

    if board_size is None:
        pygame.quit()
        return

    game = GoGame()
    game.change_board_size(board_size)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Pass
                    game.make_move('pass')
                elif event.key == pygame.K_r:  # Reset game
                    game.reset_game()
                elif event.key == pygame.K_m:  # Back to menu
                    board_size = menu.run()
                    if board_size is not None:
                        game.change_board_size(board_size)

            elif event.type == pygame.MOUSEBUTTONDOWN and not game.show_score:
                x, y = event.pos
                grid_x = round((x - game.board.margin) / CELL_SIZE)
                grid_y = round((y - game.board.margin) / CELL_SIZE)

                if 0 <= grid_x < game.board.size and 0 <= grid_y < game.board.size:
                    game.make_move((grid_x, grid_y))

        game.draw()
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()