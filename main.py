from game import GoGame
from menu import Menu
from dqn_agent import DQNAgent
import pygame
from constants import CELL_SIZE, WINDOW_SIZE, BLACK
import time
import threading
import numpy as np


def main():
    pygame.init()

    menu = Menu()
    board_size, play_with_bot, bot_vs_bot = menu.run()

    if board_size is None:
        pygame.quit()
        return

    game = GoGame()
    game.change_board_size(board_size)
    running = True

    bot1 = None
    bot2 = None
    if play_with_bot and not bot_vs_bot:
        model_path = f"bot1_model_{board_size}x{board_size}.keras"
        bot1 = DQNAgent(board_size, model_path)
    elif bot_vs_bot:
        model_path1 = f"bot1_model_{board_size}x{board_size}.keras"
        model_path2 = f"bot2_model_{board_size}x{board_size}.keras"
        bot1 = DQNAgent(board_size, model_path1)
        bot2 = DQNAgent(board_size, model_path2)

    bot_pass_message = None
    bot_pass_time = 0
    message_duration = 2.0
    move_count = 0
    max_moves = board_size * board_size * 4
    bot_delay = 0.05
    clock = pygame.time.Clock()
    bot_thinking = False

    def handle_bot_turn(bot, player):
        nonlocal bot_pass_message, bot_pass_time, move_count, bot_thinking
        if bot is None:
            return False

        bot_thinking = True
        state = game.get_state()
        state = np.reshape(state, [1, board_size, board_size, 1])

        # Thêm biến đếm thời gian và số lần thử
        start_time = time.time()
        max_attempts = board_size * board_size  # Số lần thử tối đa
        attempts = 0

        invalid_moves = []
        # Danh sách các nước đi ưu tiên (để ăn quân xâm nhập)
        priority_moves = []

        # Trước tiên, tìm các nước đi có thể ăn quân đối phương
        opponent = 'white' if player == 'black' else 'black'
        for y in range(board_size):
            for x in range(board_size):
                # Bỏ qua các ô đã có quân
                if game.board.board[y][x] is not None:
                    continue

                # Kiểm tra xem nước đi này có ăn được quân đối phương không
                can_capture = False
                game.board.board[y][x] = player  # Thử đặt quân

                # Kiểm tra các nhóm quân đối phương xung quanh
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_size and 0 <= ny < board_size and
                            game.board.board[ny][nx] == opponent):
                        group = game.board.get_group(nx, ny)
                        if game.board.count_liberties(group) == 0:
                            can_capture = True
                            break

                # Hoàn trả lại trạng thái ban đầu
                game.board.board[y][x] = None

                if can_capture:
                    # Đây là nước đi ưu tiên vì có thể ăn quân
                    action = y * board_size + x
                    priority_moves.append(action)

        # Chạy dự đoán trong luồng chính để tránh lỗi tensor
        while attempts < max_attempts and time.time() - start_time < 10.0:  # Giới hạn 10 giây
            # Nếu có nước đi ưu tiên, chọn một trong số đó
            if priority_moves:
                action = priority_moves[0]  # Chọn nước đi ưu tiên đầu tiên
                priority_moves.pop(0)
            else:
                action = bot.act(state)

            attempts += 1

            if action == board_size * board_size:  # Nếu bot chọn pass
                # Chỉ pass khi không còn nước đi ưu tiên
                if not priority_moves:
                    break
                else:
                    continue
            else:
                x, y = divmod(action, board_size)

                # Kiểm tra nước đi có hợp lệ không
                if action in invalid_moves:
                    continue  # Bỏ qua nước đi đã biết là không hợp lệ

                # Kiểm tra các điều kiện hợp lệ cơ bản
                if (x < 0 or x >= board_size or y < 0 or y >= board_size or
                        game.board.board[y][x] is not None or
                        game.would_be_suicide(x, y) or
                        game.is_ko(x, y)):
                    invalid_moves.append(action)
                    continue

                # Kiểm tra xem đây có phải là mắt (eye) không
                is_eye = True
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_size and 0 <= ny < board_size and
                            game.board.board[ny][nx] != player):
                        is_eye = False
                        break

                # Kiểm tra xem có quân đối phương xung quanh không
                has_opponent_nearby = False
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_size and 0 <= ny < board_size and
                            game.board.board[ny][nx] == opponent):
                        has_opponent_nearby = True
                        break

                # Nếu là mắt và không có quân đối phương xung quanh, bỏ qua
                if is_eye and not has_opponent_nearby:
                    invalid_moves.append(action)
                    continue

                # Kiểm tra xem đây có phải là lãnh thổ đã được bao quanh hoàn toàn không
                if game.board.territory[y][x] == player:
                    # Đếm số quân đối phương xung quanh
                    opponent_neighbors = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < board_size and 0 <= ny < board_size and
                                game.board.board[ny][nx] == opponent):
                            opponent_neighbors += 1

                    # Nếu đây là lãnh thổ bị bao vây hoàn toàn và không có quân đối phương xung quanh
                    if opponent_neighbors == 0:
                        invalid_moves.append(action)
                        continue

                # Kiểm tra xem nước đi này có ăn được quân đối phương không
                game.board.board[y][x] = player  # Thử đặt quân
                can_capture = False

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < board_size and 0 <= ny < board_size and
                            game.board.board[ny][nx] == opponent):
                        group = game.board.get_group(nx, ny)
                        if game.board.count_liberties(group) == 0:
                            can_capture = True
                            break

                # Hoàn trả lại trạng thái ban đầu
                game.board.board[y][x] = None

                # Nếu nước đi này ăn được quân đối phương, ưu tiên chọn
                if can_capture:
                    break

                # Nếu vượt qua tất cả các kiểm tra, đây là nước đi hợp lệ
                break

        # Nếu hết thời gian hoặc đã thử tất cả vị trí mà không tìm được nước đi hợp lệ
        if attempts >= max_attempts or time.time() - start_time >= 10.0 or action in invalid_moves:
            game.make_move('pass')
            bot_pass_message = f"Bot {player.capitalize()} PASSED! (timeout)"
            bot_pass_time = time.time()
        else:
            if action == board_size * board_size:
                game.make_move('pass')
                bot_pass_message = f"Bot {player.capitalize()} PASSED!"
                bot_pass_time = time.time()
            else:
                x, y = divmod(action, board_size)
                if game.make_move((x, y)):
                    bot_pass_message = None
                    move_count += 1

        next_state = game.get_state()
        state = np.reshape(state, (1, board_size, board_size, 1))
        next_state = np.reshape(next_state, (1, board_size, board_size, 1))
        reward = game.get_reward(player)
        done = game.show_score or move_count >= max_moves
        bot.remember(state, action, reward, next_state, done)

        # Huấn luyện trong luồng riêng
        def train_bot():
            bot.replay(32)
            bot_thinking = False  # Đặt lại trạng thái sau khi huấn luyện

        threading.Thread(target=train_bot).start()
        return not done

    def train_bots_async():
        if bot1 is not None:
            bot1.replay(32)
            bot1.update_target_model()
            bot1.save_model(f"bot1_model_{board_size}x{board_size}.keras")
        if bot2 is not None:
            bot2.replay(32)
            bot2.update_target_model()
            bot2.save_model(f"bot2_model_{board_size}x{board_size}.keras")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.make_move('pass')
                    move_count += 1
                    if play_with_bot and not bot_vs_bot and game.current_player == 'white' and not game.show_score:
                        threading.Thread(target=handle_bot_turn, args=(bot1, 'white')).start()
                    elif bot_vs_bot and not game.show_score:
                        if game.current_player == 'black':
                            threading.Thread(target=handle_bot_turn, args=(bot1, 'black')).start()
                        else:
                            threading.Thread(target=handle_bot_turn, args=(bot2, 'white')).start()

                elif event.key == pygame.K_r:
                    game.reset_game()
                    move_count = 0
                    bot_pass_message = None
                    bot_thinking = False

                elif event.key == pygame.K_m:
                    board_size, play_with_bot, bot_vs_bot = menu.run()
                    if board_size is not None:
                        game.change_board_size(board_size)
                        max_moves = board_size * board_size * 4
                        if play_with_bot and not bot_vs_bot:
                            model_path = f"bot1_model_{board_size}x{board_size}.keras"
                            bot1 = DQNAgent(board_size, model_path)
                            bot2 = None
                        elif bot_vs_bot:
                            model_path1 = f"bot1_model_{board_size}x{board_size}.keras"
                            model_path2 = f"bot2_model_{board_size}x{board_size}.keras"
                            bot1 = DQNAgent(board_size, model_path1)
                            bot2 = DQNAgent(board_size, model_path2)
                        else:
                            bot1 = None
                            bot2 = None
                        move_count = 0
                    bot_pass_message = None
                    bot_thinking = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not game.show_score:
                if not play_with_bot and not bot_vs_bot or (play_with_bot and game.current_player == 'black'):
                    x, y = event.pos
                    grid_x = round((x - game.board.margin) / CELL_SIZE)
                    grid_y = round((y - game.board.margin) / CELL_SIZE)
                    if 0 <= grid_x < game.board.size and 0 <= grid_y < game.board.size:
                        if game.make_move((grid_x, grid_y)):
                            move_count += 1
                            if play_with_bot and not bot_vs_bot and game.current_player == 'white':
                                threading.Thread(target=handle_bot_turn, args=(bot1, 'white')).start()

        if bot_vs_bot and not game.show_score and move_count < max_moves:
            if game.current_player == 'black':
                if not handle_bot_turn(bot1, 'black'):
                    threading.Thread(target=train_bots_async).start()
                    game.reset_game()
                    move_count = 0
            else:
                if not handle_bot_turn(bot2, 'white'):
                    threading.Thread(target=train_bots_async).start()
                    game.reset_game()
                    move_count = 0
            time.sleep(bot_delay)

        game.draw()
        move_text = game.font.render(f"Moves: {move_count}/{max_moves}", True, BLACK)
        game.screen.blit(move_text, (10, 70))


        if bot_thinking:
            thinking_text = game.font.render("Bot thinking...", True, (255, 0, 0))
            game.screen.blit(thinking_text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE - 50))

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
        clock.tick(60)

    if bot1 is not None and board_size is not None:
        bot1.save_model(f"bot1_model_{board_size}x{board_size}.keras")
    if bot2 is not None and board_size is not None:
        bot2.save_model(f"bot2_model_{board_size}x{board_size}.keras")
    pygame.quit()


if __name__ == "__main__":
    main()
