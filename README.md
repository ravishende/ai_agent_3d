# ai_agent_3d

An agent that uses policy iteration to run through a 3d Temple Run / Subway Surfers style environment

# Setup

1. Create and run virtual environment
   `python3 -m venv .venv`
   `source .venv/bin/activate`
2. Install required packages
   `python3 -m pip install -r requirements.txt`

# Run

## Running the Agent

To see the agent play, run `python3 run_rl_agent.py` with some optional command line arguments:

- Ex: `python3 run_rl_agent.py --length=15 --difficulty=hard --visualize --quiet`

- To change the output:

  - `--visualize`: Shows the agent running through the course in 3D
  - `--quiet`: Doesn't print verbose messages about the agent's process

- By default, maps will be generated randomly. Customize the map generation with the following:

  - `--length=...` (int): length of the randomly generated map
  - `--width=...` (int): width of the randomly generated map
  - `--difficulty=...` (str): difficulty use with options `easy`, `medium`, `hard`, or `expert`.

- If you prefer a pre-determined map, you can specify which map to use with `--fixed --map=...`.

  - Ex: `python3 play.py --fixed --map=3`
  - Make sure to include `--fixed` if you don't want a randomly generated map.

- To change the 3D visualization of your game, specify the following parameters

  - `--cubesize=...` (float): sidelength of the visualized cubes (default is 2.0)
  - `--spacing=...` (float): space between each slice of cubes (default is 40.0)

## Playing the Game

To be able to play through a text-based game and see it visualized, run `python3 play.py` with some optional command line arguments (any of the previously mentioned arguments in running the agent except `--quiet`).

- Ex: `python3 play.py --width=5 --length=15 --difficulty=hard`

To just play the text game, run `python3 play_text.py`

To just see the environment, run `python3 env_3d.py`
