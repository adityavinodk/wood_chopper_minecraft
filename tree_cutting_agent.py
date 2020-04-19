import argparse
import time
import random
import os
import json
from layers import Layers
from agent import Agent
try:
    import MalmoPython
except ImportError:
    print('Add MalmoPython folder to PYTHON sys.path')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

ap = argparse.ArgumentParser(
    description='Script to run the Malmo agent to cut trees')
ap.add_argument('--train', type=str2bool,
                help="boolean value whether to train or not", default=True)
ap.add_argument('--number_of_missions', type=int,
                help="number of missions for training, defaults to 100", default=100)
ap.add_argument('--learning_rate', type=float,
                help="learning rate of the Q network", default=0.001)
ap.add_argument('--epsilon', type=float,
                help="epsilon value for agent exploration during training, defaults to 0.6", default=0.6)
ap.add_argument('--alpha', type=float,
                help="alpha value for exponential update of Q value, defaults to 0.1", default=0.1)
ap.add_argument('--gamma', type=float,
                help="gamma value for partial reward based on maximum Q value of next state, defaults to 0.9", default=0.9)
ap.add_argument('--save_model_name', type=str,
                help='name of the file to save the weights after each mission, defaults to weights.npy', default='weights.npy')
ap.add_argument('--weights_file', type=str,
                help='name of the weights file, defaults to None', default='None')
arguments = vars(ap.parse_args())

agent_host = MalmoPython.AgentHost()
action_set = ["movenorth 1", "movesouth 1",
              "movewest 1", "moveeast 1", "attack 1"]
action_set_length = len(action_set)
max_retries = 3
num_missions = 150

if arguments['weights_file'] == 'None':
    model = Layers(num_classes=action_set_length,
                   batch_size=1, learning_rate=arguments['learning_rate'], save_model_name=arguments['save_model_name'])
else:
    model = Layers(num_classes=action_set_length,
                   batch_size=1, learning_rate=arguments['learning_rate'], save_model_name=arguments['weights_file'], weights_file=arguments['weights_file'])

if arguments['train']:
    epsilon_value = arguments['epsilon']
    agent = Agent(agent_host, actions=action_set,
                  epsilon=epsilon_value, alpha=arguments['alpha'], gamma=arguments['gamma'], model=model)

    for mission_count in range(arguments['number_of_missions']):
        print("\nMISSION %d\n" % (mission_count+1))

        mission_xml = agent.generateXML()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.allowAllDiscreteMovementCommands()
        my_mission.requestVideo(600, 400)
        my_mission.setViewpoint(0)

        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        print("Waiting for the mission to start ", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        print()

        print("Mission running ", end=' ')
        time.sleep(1)

        agent.train(mission_count)

        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

    print("\nMission ended")
# else:
    # TODO: Add testing segment