# Go 2D

## Introduction

This is a Go board game with a top-down 2D graphic interface, developed in Python. The game features advanced AI learning using Deep Q-Learning and supports multiple game modes.

## Key Features

- **Intuitive 2D Interface**: Clearly displays the board and stones
- **Multiple Game Modes**:
  - Human vs Human
  - Human vs AI
  - AI vs AI
- **Flexible Board Sizes**: Support for 9x9, 13x13, and 19x19 boards
- **Advanced AI**:
  - Deep Q-Learning with Double DQN and hybrid with MCTS
  - GPU-accelerated training
  - Adaptive learning strategies
- **Game Rule Enforcement**: 
  - Automatically checks valid moves
  - Calculates scores
  - Captures stones
- **AI Training**:
  - Experience replay
  - Intelligent move selection
  - Territory and liberty tracking

## System Requirements

- Python 3.8 or later
- Required libraries:
  ```
  pygame
  numpy
  tensorflow
  threading
  ```

## Installation

1. Clone the repository
2. Install required libraries:
   ```sh
   pip install pygame numpy tensorflow
   ```

## Usage Guide

### Run the Game

```sh
python main.py
```

### How to Play

- Select game mode from the menu
- Choose board size
- Place stones by clicking on empty intersections
- Use special keys:
  - R: Reset game
  - M: Return to menu
  - Space: Pass turn

### Game Modes

- **1v1**: Two human players take turns
- **Human vs AI**: Play against an intelligent bot
- **AI vs AI**: Watch two AI players compete

## AI Features

- Adaptive learning algorithm
- Intelligent move prioritization
- GPU-accelerated training
- Persistent model saving/loading

## Project Structure

- `main.py`: Main game entry point
- `game.py`: Core game logic
- `board.py`: Board representation
- `dqn_agent.py`: AI learning agent
- `menu.py`: Game menu interface

## Future Improvements

- Online multiplayer support
- Enhanced AI training techniques
- More detailed game analytics
- Additional difficulty levels

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

