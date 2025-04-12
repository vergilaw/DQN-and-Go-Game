from game import GoGame
from menu import Menu
from dqn_agent import DQNAgent
import pygame
from constants import CELL_SIZE, WINDOW_SIZE, BLACK
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import traceback

def main():
    thread_pool = None
    training_future = None
    stop_training = False
    bot1 = None
    bot2 = None
    board_size = None
    pygame_initialized = False
    active_threads = []  # Danh sách để lưu các thread đang chạy

    try:
        thread_pool = ThreadPoolExecutor(max_workers=1)
        training_future = None
        stop_training = False

        pygame.init()
        pygame_initialized = True
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Cờ Vây - Go Game')
        font = pygame.font.Font(None, 36)

        menu = Menu()
        board_size, play_with_bot, bot_vs_bot = menu.run()

        if board_size is None:
            return

        game = GoGame()
        game.change_board_size(board_size)
        running = True

        if play_with_bot and not bot_vs_bot:
            model_path = f"bot1_model_{board_size}x{board_size}.keras"
            bot1 = DQNAgent(board_size, model_path, use_mcts=True, mcts_simulations=50)
        elif bot_vs_bot:
            model_path1 = f"bot1_model_{board_size}x{board_size}.keras"
            model_path2 = f"bot2_model_{board_size}x{board_size}.keras"
            bot1 = DQNAgent(board_size, model_path1, use_mcts=True, mcts_simulations=50)
            bot2 = DQNAgent(board_size, model_path2, use_mcts=True, mcts_simulations=50)

        bot_pass_message = None
        bot_pass_time = 0
        message_duration = 2.0
        move_count = 0
        max_moves = board_size * board_size * 4
        bot_delay = 0.5
        clock = pygame.time.Clock()
        bot_thinking = False
        score_display_time = None

        def handle_bot_turn(bot, player):
            nonlocal bot_pass_message, bot_pass_time, move_count, bot_thinking, training_future, stop_training
            if bot is None or stop_training:
                return False

            bot_thinking = True
            state = game.get_state()
            state = np.reshape(state, [1, board_size, board_size, 1])

            start_time = time.time()
            max_attempts = board_size * board_size
            attempts = 0
            invalid_moves = []
            priority_moves = []

            opponent = 'white' if player == 'black' else 'black'
            for y in range(board_size):
                for x in range(board_size):
                    if game.board.board[y][x] is not None:
                        continue
                    can_capture = False
                    game.board.board[y][x] = player
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < board_size and 0 <= ny < board_size and
                                game.board.board[ny][nx] == opponent):
                            group = game.board.get_group(nx, ny)
                            if game.board.count_liberties(group) == 0:
                                can_capture = True
                                break
                    game.board.board[y][x] = None
                    if can_capture:
                        action = y * board_size + x
                        priority_moves.append(action)

            action = None  # Khởi tạo action
            while attempts < max_attempts and time.time() - start_time < 10.0:
                try:
                    if priority_moves:
                        action = priority_moves[0]
                        priority_moves.pop(0)
                    else:
                        action = bot.act(state, game=game)

                    attempts += 1
                    if action is None:  # Kiểm tra action
                        invalid_moves.append(action)
                        continue
                    if action == board_size * board_size:
                        if not priority_moves:
                            break
                        else:
                            continue
                    else:
                        x, y = divmod(action, board_size)
                        if action in invalid_moves:
                            continue
                        if (x < 0 or x >= board_size or y < 0 or y >= board_size or
                                game.board.board[y][x] is not None or
                                game.would_be_suicide(x, y) or
                                game.is_ko(x, y)):
                            invalid_moves.append(action)
                            continue

                        is_eye = True
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < board_size and 0 <= ny < board_size and
                                    game.board.board[ny][nx] != player):
                                is_eye = False
                                break

                        has_opponent_nearby = False
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < board_size and 0 <= ny < board_size and
                                    game.board.board[ny][nx] == opponent):
                                has_opponent_nearby = True
                                break

                        if is_eye and not has_opponent_nearby:
                            invalid_moves.append(action)
                            continue

                        if game.board.territory[y][x] == player:
                            opponent_neighbors = 0
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < board_size and 0 <= ny < board_size and
                                        game.board.board[ny][nx] == opponent):
                                    opponent_neighbors += 1
                            if opponent_neighbors == 0:
                                invalid_moves.append(action)
                                continue

                        game.board.board[y][x] = player
                        can_capture = False
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < board_size and 0 <= ny < board_size and
                                    game.board.board[ny][nx] == opponent):
                                group = game.board.get_group(nx, ny)
                                if game.board.count_liberties(group) == 0:
                                    can_capture = True
                                    break
                        game.board.board[y][x] = None

                        if can_capture:
                            break
                        break
                except Exception as e:
                    print(f"Error during bot move evaluation: {e}")
                    if 'action' in locals() and action is not None and action not in invalid_moves:
                        invalid_moves.append(action)
                    continue

            done = False  # Khởi tạo mặc định
            try:
                if attempts >= max_attempts or time.time() - start_time >= 10.0 or action in invalid_moves or action is None:
                    game.make_move('pass')
                    bot_pass_message = f"Bot {player.capitalize()} PASSED! (timeout or invalid action)"
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
                next_state = np.reshape(next_state, [1, board_size, board_size, 1])
                reward = game.get_reward(player)
                done = game.show_score or move_count >= max_moves

                print(f"Storing to memory: state type: {type(state)}, next_state type: {type(next_state)}, "
                      f"action type: {type(action)}, reward type: {type(reward)}, done type: {type(done)}")
                bot.remember(state, action, reward, next_state, done)

                def train_bot():
                    nonlocal bot_thinking
                    try:
                        if not stop_training:
                            bot.replay(32)
                    except Exception as e:
                        print(f"Training error: {e}")
                    bot_thinking = False

                if not training_future or training_future.done():
                    if not stop_training:
                        training_future = thread_pool.submit(train_bot)

            except Exception as e:
                print(f"Error during bot move execution: {e}")
                try:
                    game.make_move('pass')
                    bot_pass_message = f"Bot {player.capitalize()} PASSED! (error)"
                    bot_pass_time = time.time()
                except:
                    pass

            bot_thinking = False
            return not done

        def train_bots_async():
            try:
                if bot1 is not None:
                    bot1.replay(32)
                    bot1.update_target_model()
                    bot1.save(f"bot1_model_{board_size}x{board_size}.keras")
                if bot2 is not None:
                    bot2.replay(32)
                    bot2.update_target_model()
                    bot2.save(f"bot2_model_{board_size}x{board_size}.keras")
            except Exception as e:
                print(f"Error during async training: {e}")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.make_move('pass')
                        move_count += 1
                        if play_with_bot and not bot_vs_bot and game.current_player == 'white' and not game.show_score:
                            thread = threading.Thread(target=handle_bot_turn, args=(bot1, 'white'))
                            thread.start()
                            active_threads.append(thread)
                        elif bot_vs_bot and not game.show_score and not bot_thinking:
                            if game.current_player == 'black':
                                thread = threading.Thread(target=handle_bot_turn, args=(bot1, 'black'))
                                thread.start()
                                active_threads.append(thread)
                            else:
                                thread = threading.Thread(target=handle_bot_turn, args=(bot2, 'white'))
                                thread.start()
                                active_threads.append(thread)

                    elif event.key == pygame.K_r:
                        game.reset_game()
                        move_count = 0
                        bot_pass_message = None
                        bot_thinking = False
                        score_display_time = None

                    elif event.key == pygame.K_m:
                        # Dừng các thread trước khi đổi map
                        stop_training = True
                        for thread in active_threads:
                            if thread.is_alive():
                                thread.join()  # Chờ thread hoàn thành
                        active_threads.clear()  # Xóa danh sách thread

                        board_size, play_with_bot, bot_vs_bot = menu.run()
                        if board_size is not None:
                            game.change_board_size(board_size)
                            max_moves = board_size * board_size * 4
                            if play_with_bot and not bot_vs_bot:
                                model_path = f"bot1_model_{board_size}x{board_size}.keras"
                                bot1 = DQNAgent(board_size, model_path, use_mcts=True, mcts_simulations=50)
                                bot2 = None
                            elif bot_vs_bot:
                                model_path1 = f"bot1_model_{board_size}x{board_size}.keras"
                                model_path2 = f"bot2_model_{board_size}x{board_size}.keras"
                                bot1 = DQNAgent(board_size, model_path1, use_mcts=True, mcts_simulations=50)
                                bot2 = DQNAgent(board_size, model_path2, use_mcts=True, mcts_simulations=50)
                            else:
                                bot1 = None
                                bot2 = None
                            move_count = 0
                            bot_pass_message = None
                            bot_thinking = False
                            score_display_time = None
                            stop_training = False  # Reset cờ để tiếp tục huấn luyện

                elif event.type == pygame.MOUSEBUTTONDOWN and not game.show_score:
                    if not play_with_bot and not bot_vs_bot or (play_with_bot and game.current_player == 'black'):
                        x, y = event.pos
                        grid_x = round((x - game.board.margin) / CELL_SIZE)
                        grid_y = round((y - game.board.margin) / CELL_SIZE)
                        if 0 <= grid_x < game.board.size and 0 <= grid_y < game.board.size:
                            if game.make_move((grid_x, grid_y)):
                                move_count += 1
                                if play_with_bot and not bot_vs_bot and game.current_player == 'white':
                                    thread = threading.Thread(target=handle_bot_turn, args=(bot1, 'white'))
                                    thread.start()
                                    active_threads.append(thread)

            if bot_vs_bot and not game.show_score and move_count < max_moves and not bot_thinking:
                if game.current_player == 'black':
                    thread = threading.Thread(target=handle_bot_turn, args=(bot1, 'black'))
                    thread.start()
                    active_threads.append(thread)
                else:
                    thread = threading.Thread(target=handle_bot_turn, args=(bot2, 'white'))
                    thread.start()
                    active_threads.append(thread)
                time.sleep(bot_delay)

            if bot_vs_bot and game.show_score:
                if score_display_time is None:
                    score_display_time = time.time()
                current_time = time.time()
                if current_time - score_display_time >= 5:
                    print("Game ended. Training bots and starting new game.")
                    threading.Thread(target=train_bots_async).start()
                    game.reset_game()
                    move_count = 0
                    score_display_time = None
                    bot_thinking = False
                else:
                    remaining = 5 - (current_time - score_display_time)
                    next_game_text = font.render(f"Next game in {remaining:.1f}s...", True, BLACK)
                    screen.blit(next_game_text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE - 30))

            game.draw(screen, font)
            move_text = font.render(f"Moves: {move_count}/{max_moves}", True, BLACK)
            screen.blit(move_text, (10, 70))

            if bot_thinking:
                thinking_text = font.render("Bot thinking...", True, (255, 0, 0))
                screen.blit(thinking_text, (WINDOW_SIZE // 2 - 100, WINDOW_SIZE - 50))

            if bot_pass_message and time.time() - bot_pass_time < message_duration:
                pass_font = pygame.font.Font(None, 48)
                pass_text = pass_font.render(bot_pass_message, True, (255, 0, 0))
                text_rect = pass_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 100))
                pygame.draw.rect(screen, (255, 255, 200),
                                 (text_rect.x - 10, text_rect.y - 10,
                                  text_rect.width + 20, text_rect.height + 20))
                pygame.draw.rect(screen, (0, 0, 0),
                                 (text_rect.x - 10, text_rect.y - 10,
                                  text_rect.width + 20, text_rect.height + 20), 2)
                screen.blit(pass_text, text_rect)
            else:
                bot_pass_message = None

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        print("Game interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    finally:
        stop_training = True
        for thread in active_threads:
            if thread.is_alive():
                thread.join()
        active_threads.clear()

        if thread_pool:
            print("Shutting down thread pool...")
            thread_pool.shutdown(wait=True)
        try:
            if bot1 is not None and board_size is not None:
                print(f"Saving bot1 model for {board_size}x{board_size} board...")
                bot1.save(f"bot1_model_{board_size}x{board_size}.keras")
                bot1.plot_training_progress(bot_name="bot1")
            if bot2 is not None and board_size is not None:
                print(f"Saving bot2 model for {board_size}x{board_size} board...")
                bot2.save(f"bot2_model_{board_size}x{board_size}.keras")
                bot2.plot_training_progress(bot_name="bot2")
        except Exception as e:
            print(f"Error saving models or plotting: {e}")
        if pygame_initialized:
            print("Quitting pygame...")
            pygame.quit()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()