# ai_agent_3d

An agent that uses policy iteration to run through a 3d Temple Run / Subway Surfers style environment

# Setup

1. Create and run virtual environment
   `python3 -m venv .venv`
   `source .venv/bin/activate`
2. Install required packages
   `python3 -m pip install -r requirements.txt`

# Run

Project is still in progress.

- To be able to play through a text-based game and see it visualized, run `python3 play.py` with some optional command line arguments.

  - Ex: `python3 play.py --width=5 --length=15 --difficulty=hard`

  - By default, maps will be generated randomly. Customize the map generation with the following:

    - `--length=...` (int): length of the randomly generated map
    - `--width=...` (int): width of the randomly generated map
    - `--difficulty=...` (str): difficulty use with options `easy`, `medium`, `hard`, or `expert`.

  - You can specify which predetermined map to use with `--fixed --map=...`.

    - Ex: `python3 play.py --fixed --map=3`
    - Make sure to include `--fixed` if you don't want a randomly generated map.

  - To change the 3D visualization of your game, specify the following parameters

    - `--cubesize=...` (float): sidelength of the visualized cubes (default is 2.0)
    - `--spacing=...` (float): space between each slice of cubes (default is 40.0)

- To just play the text game, run `python3 play_text.py`

- To just see the environment, run `python3 env_3d.py`
