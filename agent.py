import numpy as np
import json
import tensorflow as tf
import random
import time
from PIL import Image

tf.compat.v1.enable_eager_execution()
save_images = True

class Agent:
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, agent_host, actions, model, epsilon=0.8, alpha=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = True
        self.agent = agent_host
        self.image_target_size = (224,224)
        self.actions = actions
        self.model = model
    
    def generateXML(self):
        trees = list()
        for i in range(3):
            newtree = (random.randint(1,9), random.randint(5,19))
            while newtree in trees:
                newtree = (random.randint(1,9), random.randint(5,19))
            trees.append(newtree)

        mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

            <About>
                <Summary>Tree Cutting Mission</Summary>
            </About>

            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator generatorString="3;7,2*3,2;4;" />
                    <DrawingDecorator>
                        <DrawCuboid x1="0" y1="46" z1="0" x2="10" y2="50" z2="25" type="air" />            <!-- limits of our arena -->
                        <DrawCuboid x1="0" y1="45" z1="0" x2="10" y2="45" z2="25" type="grass" />           <!-- dirt floor -->
                        
                        <DrawCuboid x1="0" y1="46" z1="0" x2="0" y2="48" z2="25" type="stone" />
                        <DrawCuboid x1="10" y1="46" z1="0" x2="10" y2="48" z2="25" type="stone" />
                        <DrawCuboid x1="0" y1="46" z1="25" x2="10" y2="48" z2="25" type="stone" />
                        <DrawCuboid x1="0" y1="46" z1="0" x2="10" y2="48" z2="0" type="stone" />

                        <DrawBlock   x="4"   y="45"  z="1"  type="cobblestone" />                           <!-- the starting marker -->
                        
                        <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="45"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="46"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- another destination marker -->
                        <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="47"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="48"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- another destination marker -->

                        <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="45"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="46"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- another destination marker -->
                        <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="47"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="48"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- another destination marker -->

                        <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="45"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="46"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- another destination marker -->
                        <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="47"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- the destination marker -->
                        <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="48"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- another destination marker -->
                    </DrawingDecorator>
                    <ServerQuitFromTimeUp timeLimitMs="300000" />
                    <ServerQuitWhenAnyAgentFinishes />
                </ServerHandlers>
            </ServerSection>

            <AgentSection mode="Survival">
                <Name>Barbie</Name>
                <AgentStart>
                    <Placement x="4.5" y="46.0" z="1.5" pitch="0" yaw="0"/>
                    <Inventory>
                        <InventoryItem slot="0" type="diamond_axe"/>
                    </Inventory>
                    <!-- <Placement x="1" y="45" z="0" pitch="0" yaw="0" /> -->
                </AgentStart>
                <AgentHandlers>
                    <ObservationFromFullStats />
                    <DiscreteMovementCommands />
                    <VideoProducer want_depth="false">
                        <Width>640</Width>
                        <Height>480</Height>
                    </VideoProducer>
                    <RewardForSendingCommand reward="-1" />
                    <RewardForTouchingBlockType>
                        <Block reward="100.0" type="log" behaviour="oncePerBlock" />
                        <Block reward="50.0" type="leaves" behaviour="oncePerBlock" />
                    </RewardForTouchingBlockType>
                    <AgentQuitFromTimeUp timeLimitMs="300000" />
                </AgentHandlers>
            </AgentSection>
        </Mission>
        '''
        return mission_xml
        
    def get_state(self, world_state):
        while len(world_state.video_frames) == 0:
            world_state = self.agent.peekWorldState()
        frame = world_state.video_frames[-1]
        image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
        image = image.resize(size=self.image_target_size, resample=Image.LANCZOS)
        np_image = np.asarray(image)
        np_image = np_image/255.0
        np_image = np_image[np.newaxis, :, :, :]

        return np_image

    def train(self, mission_count):
        ''' Take model and train agent get state frame, 
            predict the Q values of the state using the model,
            choose an action using the greedy epsilon value, 
            store the next state as as part of a temporary buffer,
            randomly select previous instances in the buffer and
            perform the q value update function - experience replay step,
            set y as q values along with updated q value for action, 
            train the model with original state and y, 
            redo all steps for number of epochs 
            
            Perform the approach similar to the algorithm stated in DQN with Atari Paper 
        '''
        
        experiences = list()
        count = 0
        world_state = self.agent.getWorldState()

        while world_state.is_mission_running:
            print("\n------------\nMISSION", mission_count, "RUN", count, "\n------------\n")
            world_state = self.agent.getWorldState()
            state = self.get_state(world_state) # Gets current state
            qvals = self.model.predict(state).numpy() # Gets prediction of q values for current state
            print('Got Q values of current state:', qvals)

            # Picks a random action OR picks best action based on random probability
            if self.epsilon>0.3 or random.random() < self.epsilon: 
                action = random.randint(0,len(self.actions)-1)
                print('Choosing random action ', self.actions[action])
            else:
                action = np.argmax(qvals)
                print('Choosing best action ', self.actions[action])
            time.sleep(1)

            self.agent.sendCommand(self.actions[action])
            world_state = self.agent.getWorldState() # Gets new world state

            new_state = self.get_state(world_state)
            reward = sum(r.getValue() for r in world_state.rewards) # Fetches reward for the state
            print('reward at current state:', reward)
            experiences.append([state, action, reward, new_state]) # Appends (s, a, r, s')
            print('Saved the experience of the agent')
            # Experience replay part
            # Finds a random number of random (s, a, r, s') sets from d
            randindexes = random.sample(range(0, len(experiences)), random.randint(0, len(experiences)-1))
            print('Random indexes chosen to update:', randindexes)
            for i in randindexes:
                reward = experiences[i][2]
                action_index = experiences[i][1]
                qval = self.model.predict( experiences[i][0] ).numpy() # finds predicted q values of s ( d[i][0] )
                newQ = self.model.predict( experiences[i][3] ).numpy() # finds predicted q values of s' ( d[i][3] )
                maxQ = np.max(newQ)
                print('Got Q values of next state with maxQ:', maxQ)
                
                y = np.zeros((1, len(self.actions))) # stores output y after updating
                y[:] = qval[:]

                # if experiences[i][2] == -1: # non-terminal state
                print('original qval for action:', qval[0][action_index])
                update = (1 - self.alpha) * qval[0][action_index] + self.alpha * (reward + self.gamma * maxQ)
                print('Changes by', update - qval[0][action_index])
                # else: # terminal state
                    # update = (1 - self.alpha) * qval[0][experiences[i][1]] + self.alpha * (experiences[i][2])
                
                y[0][action_index] = update
                self.model.train_step(state, tf.convert_to_tensor(y))
                print('Model trained for experience ', i)
            world_state = self.agent.getWorldState()
            count+=1