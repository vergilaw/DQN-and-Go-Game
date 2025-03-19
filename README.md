# Go 2D

## Introduction

This is a Go board game with a top-down 2D graphic interface, developed in Python. The game supports:

- Playing against a bot using Q-learning
- 1v1 mode for two human players.

## Key Features

- **Intuitive 2D Interface**: Clearly displays the board and stones.
- **Various Game Modes**:
  - Play against Bot
  - Play against another human on the same device.
- **Game Rule Enforcement**: Automatically checks valid moves, calculates scores, and captures stones.
- **Save and Load Game**: Allows resuming unfinished games.

## Installation

### System Requirements

- Python 3.8 or later
- Required libraries:
  ```sh
  pygame numpy time threading tensorflow os deepcopy 
  ```

## Usage Guide

### Run the Game

Open terminal/cmd and execute:

```sh
python main.py
```

### How to Play

- **1v1 Mode**: Two players take turns placing black and white stones, chose size before play.
- **BOT Mode**: Select BOT before size and start the game.
- Click on an empty space to place a stone.
- The game ends when no more moves are possible or when both players pass their turns consecutively.
- Button:
  - R - reset
  - M - menu
  - Space - pass

