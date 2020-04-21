import time
from layers import Layers
from agent import Agent
from utils import generate_xml, read_xml_file
try:
    import MalmoPython
except ImportError:
    print('Add MalmoPython module folder to PYTHON sys.path')

class TreeCuttingAgent:
    def __init__(self, arguments):
        self.agent_host = MalmoPython.AgentHost()
        # Defined action set of the agent
        self.action_set = ["movenorth 1", "movesouth 1",
                           "movewest 1", "moveeast 1", "attack 1"]
        self.action_set_length = len(self.action_set)
        self.max_retries = 3
        # Retrieve arguments from run.py
        self.arguments = arguments

        # Depending on whether weights should be uploaded from existing file, initialize layers class
        if not arguments['weights_file']:
            self.model = Layers(num_classes=self.action_set_length,
                                learning_rate=arguments['learning_rate'], save_model_name=arguments['save_model_name'])
        else:
            self.model = Layers(num_classes=self.action_set_length,
                                learning_rate=arguments['learning_rate'], save_model_name=arguments['weights_file'], weights_file=arguments['weights_file'])
            print('\nREADING WEIGHTS FROM %s\n' % arguments['weights_file'])

    def train(self):
        # Training function
        print('----------------------------------------------------------------')
        print('TRAINING Mode')
        print('NUMBER OF MISSIONS:', self.arguments['number_of_missions'])
        print('LEARNING RATE:', self.arguments['learning_rate'])
        print('EPSILON VALUE FOR EXPLORATION:', self.arguments['epsilon'])
        print('ALPHA VALUE FOR EXPONENTIAL Q UPDATE:', self.arguments['alpha'])
        print('GAMMA VALUE FOR PARTIAL REWARD TO AGENT:',
              self.arguments['gamma'])
        print('MISSION TIME OF AGENT:', self.arguments['mission_time'])
        print('----------------------------------------------------------------')

        # generate Agent class object
        agent = Agent(self.agent_host, actions=self.action_set, batch_size=self.arguments['batch_size'], epsilon=self.arguments['epsilon'],
                      alpha=self.arguments['alpha'], gamma=self.arguments['gamma'], model=self.model, explore=self.arguments['explore'])

        # Run mission several times 
        for mission_count in range(self.arguments['number_of_missions']):

            print("\nMISSION %d\n" % (mission_count+1))
            # Generate random or specific Malmo Environment
            if not self.arguments['xml_file']:
                mission_xml = generate_xml(self.arguments['mission_time'])
                print('GENERATING RANDOM MISSION ENVIRONMENT')
            else:
                mission_xml = read_xml_file(self.arguments['xml_file'])
                print('GENERATING MISSION ENVIRONMENT FROM',
                      self.arguments['xml_file'])

            # Initiate Malmo Mission
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission_record = MalmoPython.MissionRecordSpec()
            my_mission.allowAllDiscreteMovementCommands()
            my_mission.requestVideo(600, 400)
            my_mission.setViewpoint(0)

            for retry in range(self.max_retries):
                try:
                    self.agent_host.startMission(my_mission, my_mission_record)
                    break
                except RuntimeError as e:
                    if retry == self.max_retries - 1:
                        print("Error starting mission:", e)
                        exit(1)
                    else:
                        time.sleep(2)

            print("Waiting for the mission to start ", end=' ')
            world_state = self.agent_host.getWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = self.agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)
            print()

            print("Mission running ", end=' ')
            time.sleep(1)

            # Train the agent
            agent.train(mission_count)

            while world_state.is_mission_running:
                print(".", end="")
                time.sleep(0.1)
                world_state = self.agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:", error.text)

        print("\nMISSION ENDED")

    def test(self):
        agent = Agent(self.agent_host, actions=self.action_set,
                      model=self.model)

        # Generate random or specific Malmo Environment
        if not self.arguments['xml_file']:
            mission_xml = generate_xml(self.arguments['mission_time'])
            print('GENERATING RANDOM MISSION ENVIRONMENT')
        else:
            mission_xml = read_xml_file(self.arguments['xml_file'])
            print('GENERATING MISSION ENVIRONMENT FROM',
                  self.arguments['xml_file'])

        # Initiate Malmo Mission
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.allowAllDiscreteMovementCommands()
        my_mission.requestVideo(600, 400)
        my_mission.setViewpoint(0)

        for retry in range(self.max_retries):
            try:
                self.agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == self.max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        print("Waiting for the mission to start ", end=' ')
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        print()

        print("Mission running ", end=' ')
        time.sleep(1)

        # Test the agent with uploaded weights file
        reward = agent.test()

        while world_state.is_mission_running:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)

        print("\nMISSION ENDED")
        print("\nFINAL REWARD:", reward)
