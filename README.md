# AI Agent: Surviving a 3D Obstacle Course

An agent that uses policy iteration to run through a 3D Temple Run / Subway Surfers style environment

To see some of the thought process behind it, look in `docs/`

## Setup

This assumes you have already cloned this repo. If not, run `git clone https://github.com/ravishende/ai_agent_3d.git; cd ai_agent_3d` first.

1. Create and run virtual environment
   `python3 -m venv .venv`
   `source .venv/bin/activate`
2. Install required packages
   `python3 -m pip install -r requirements.txt`

## Run

### Running the Agent

To see the agent play, run `python3 run_rl_agent.py` with some optional command line arguments:

- Ex: `python3 run_rl_agent.py --length=15 --width=9 --difficulty=expert --visualize --quiet`

- To change the output:

  - `--visualize`: Shows the agent running through the course in 3D
  - `--quiet`: Doesn't print verbose messages about the agent's process

- By default, maps will be generated randomly. Customize the map generation with the following:

  - `--length=...` (int): length of the randomly generated map
  - `--width=...` (int): width of the randomly generated map
  - `--difficulty=...` (str): difficulty use with options `easy`, `medium`, `hard`, or `expert`.
  - `--trap_spawn_prob=...` (float): frequency of traps in the map (from 0 to 1) - default is 0.3
  - `--trap_death_prob=...` (float): probability of a trap killing the player upon being activated (from 0 to 1) - default is 0.3

- If you prefer a pre-determined map, you can specify which map to use with `--fixed --map=...`.

  - Ex: `python3 play.py --fixed --map=3`
  - Make sure to include `--fixed` if you don't want a randomly generated map.

- To change the 3D visualization of your game, specify the following parameters

  - `--cubesize=...` (float): sidelength of the visualized cubes (default is 2.0)
  - `--spacing=...` (float): space between each slice of cubes (default is 40.0)

#### A Note on Traps:

Traps add a nondeterministic element to the game. If a player ends up in the same column as a trap, the trap has a certain probability of ending the game. To disable traps, set `--trap_spawn_prob=0`.

While any randomly generated map is guaranteed to have a solution that avoids obstacles, it does not guarantee that the player can avoid all traps as well. if `trap_spawn_prob=1`, every grid that has more than one solution column will contain a trap. Grids with only one solution column will never contain a trap.

Traps add complexity to the game, forcing the agent to make informed decisions on whether to risk an early end for the chance of having a longer safe path in the future. If traps were entirely avoidable, they could be treated as obstacles, rendering them useless in terms of agent decision making.

### Playing the Game

To be able to play through a text-based game and see it visualized, run `python3 play.py` with some optional command line arguments (any of the previously mentioned arguments in running the agent except `--quiet`).

- Ex: `python3 play.py --width=5 --length=15 --difficulty=hard`

To just play the text game, run `python3 play_text.py`

To just see the environment, run `python3 env_3d.py`
