import argparse
from tree_cutting_agent import TreeCuttingAgent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

# Run $python run.py -h to print the necessary arguments and their use

ap = argparse.ArgumentParser(
    description='Script to run the Malmo agent to cut trees')
ap.add_argument('--train', type=str2bool,
                help="boolean value whether to train or not", default=True)
ap.add_argument('--number_of_missions', type=int,
                help="number of missions for training, defaults to 3", default=3)
ap.add_argument('--learning_rate', type=float,
                help="learning rate of the Q network", default=0.001)
ap.add_argument('--epsilon', type=float,
                help="epsilon value for agent exploration during training, defaults to 0.6", default=0.6)
ap.add_argument('--alpha', type=float,
                help="alpha value for exponential update of Q value, defaults to 0.1", default=0.1)
ap.add_argument('--gamma', type=float,
                help="gamma value for partial reward based on maximum Q value of next state, defaults to 0.9", default=0.9)
ap.add_argument('--xml_file', type=str,
                help="path to xml file containing environment, defaults to None")
ap.add_argument('--mission_time', type=float,
                help='mission time of the agent in minutes, defaults to 2 minutes', default=2)
ap.add_argument('--save_model_name', type=str,
                help='name of the file to save the weights after each mission, defaults to weights.npy', default='weights.npy')
ap.add_argument('--weights_file', type=str,
                help='name of the weights .npy file inside weights directory, defaults to None')
ap.add_argument('--batch_size', type=int,
                help='number of frames to train in one epoch, defaults to 1', default=1)
ap.add_argument('--explore', type=str2bool,
                help="set to true if the agent should explore its surroundings, defaults to False", default=False)
arguments = vars(ap.parse_args())

if __name__ == "__main__":
    agent = TreeCuttingAgent(arguments=arguments)
    if arguments['train']:
        agent.train()
    else:
        if arguments['weights_file']:
            agent.test()
        else:
            print('Weights file not provided.')
            exit()
