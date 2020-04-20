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
                current_state = experiences[i][0]
                next_state = experiences[i][3]
                reward = experiences[i][2]
                action_index = experiences[i][1]
                qval = self.model.predict(current_state).numpy() # finds predicted q values of s ( d[i][0] )
                newQ = self.model.predict(next_state).numpy() # finds predicted q values of s' ( d[i][3] )
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
                current_loss = self.model.train_step(current_state, tf.convert_to_tensor(y))
                print('Model trained for experience ', i)

            world_state = self.agent.getWorldState()
            count+=1
