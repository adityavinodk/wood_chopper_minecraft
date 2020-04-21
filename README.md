# Artificially Intelligent Tree Cutting Agent in Minecraft
## Setup
1. Clone the repository

2. Install all requirements using pip
```sh
$ pip install -r requirements.txt
```
3. Start the Malmo Client. Visit [Microsoft Malmo Documentation](https://github.com/microsoft/malmo) for setup information.

4. Check the arguments for `tree_cutting_agent.py` using help argument and run according.
```sh
$ python run.py -h
``` 
To train the agent run `python run.py` and to test run `python run.py --train false weights_file <weights>`