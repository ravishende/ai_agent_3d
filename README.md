# ai_agent_3d

An agent that uses policy iteration to run through a 3d Temple Run / Subway Surfers style environment

# Setup

1. Create and run virtual environment
   `python3 -m venv .env`
   `source .env/bin/activate`
2. Install required packages
   `python3 -m pip install -r requirements.txt`

# Run

Project is still in progress.

- To be able to play through a text-based game and see it visualized, run `python3 play.py` with some optional command line arguments.

  - Ex: `python3 play.py --length=15 --difficulty=hard`

  - By default, maps will be generated randomly.

    - To specify the length of the randomly generated map, use `--length=...` and specify an int.
    - To specify the difficulty of the randomly genereated map, use `--difficulty=...` with options `easy`, `medium`, or `hard`.

  - You can specify which predetermined map to use with `--fixed --map=...`.
    - Ex: `python3 play.py --fixed --map=3`
    - Make sure to include `--fixed` if you don't want a randomly generated map.

- To just play the text game, run `python3 play_text.py`

- To just see the environment, run `python3 env_3d.py`
