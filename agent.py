import numpy as np
import copy
import json
import tensorflow as tf
import random
import time
from PIL import Image

tf.compat.v1.enable_eager_execution()
save_images = True

class Agent:
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, agent_host, actions, model, epsilon=0.8, alpha=0.1, gamma=0.9, batch_size=1, explore=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.agent = agent_host
        self.image_target_size = (224,224)
        self.actions = actions
        self.model = model
        self.explore = explore
        self.second_model_set = False
        self.update_count = 0
        if explore:
            self.action_weights = [0.15, 0.15, 0.15, 0.15, 0.4]
        else:
            self.action_weights = [0.2]*5

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

    def double_update_q_network(self, indexes):
        if not self.second_model_set:
            self.second_model = copy.deepcopy(self.model) 
            self.second_model.model_name = 'B'
            self.second_model_set = True

        if self.update_count % 2:
            model1 = self.model
            model2 = self.second_model
        else:
            model2 = self.model
            model1 = self.second_model
        print('%s is model1, %s is model2'%(model1.model_name, model2.model_name))
        
        X = np.zeros(shape=(len(indexes), 224, 224, 3), dtype=np.float16)
        Y = np.zeros(shape=(len(indexes), len(self.actions)), dtype=np.float16)

        for index_count in range(len(indexes)):
            i = indexes[index_count]
            current_state = self.experiences[i][0]
            next_state = self.experiences[i][3]
            reward = self.experiences[i][2]
            action_index = self.experiences[i][1]
            qVal = model1.predict(current_state).numpy()
            nextqVal = model1.predict(next_state).numpy()
            nextqVal_in_model2 = model2.predict(next_state).numpy()
            best_action_in_next_state = np.argmax(nextqVal)
            maxQ = nextqVal_in_model2[0][best_action_in_next_state] 

            X[index_count] = current_state
            y = np.zeros(shape = (1, len(self.actions)))
            y[:] = qVal[:]
            y[0][action_index] = (1 - self.alpha) * qVal[0][action_index] + self.alpha * (reward + self.gamma * maxQ)
            Y[index_count] = y

        current_loss = model1.train_step(X, Y)
        if self.update_count % 2:
            if not self.explore: model1.save_weights(self.explore)
            print('model trained for %d experiences' % len(indexes))
        self.update_count += 1

    def update_q_network(self, indexes):
        X = np.zeros(shape=(len(indexes), 224, 224, 3), dtype=np.float16)
        Y = np.zeros(shape=(len(indexes), len(self.actions)), dtype=np.float16)

        for index_count in range(len(indexes)):
            i = indexes[index_count]
            current_state = self.experiences[i][0]
            next_state = self.experiences[i][3]
            reward = self.experiences[i][2]
            action_index = self.experiences[i][1]
            qVal = self.model.predict(current_state).numpy()
            nextqVal = self.model.predict(next_state).numpy()
            maxQ = np.max(nextqVal)

            X[index_count] = current_state
            y = np.zeros(shape = (1, len(self.actions)))
            y[:] = qVal[:]
            y[0][action_index] = (1 - self.alpha) * qVal[0][action_index] + self.alpha * (reward + self.gamma * maxQ)
            Y[index_count] = y

        current_loss = self.model.train_step(X, Y)
        if not self.explore: self.model.save_weights(self.explore)

        self.update_count += 1
        print('model trained for %d experiences' % len(indexes))
        
    def train(self, mission_count):
        ''' take model and train agent get state frame, 
            predict the q values of the state using the model,
            choose an action using the greedy epsilon value, 
            store the next state as as part of a temporary buffer,
            randomly select previous instances in the buffer and
            perform the q value update function - experience replay step,
            set y as q values along with updated q value for action, 
            train the model with original state and y, 
            redo all steps for number of epochs 
            
            perform the approach similar to the algorithm stated in dqn with atari paper 
        '''
        
        self.experiences = []
        count = 0
        world_state = self.agent.getWorldState()

        while world_state.is_mission_running:
            print("\n------------\nMISSION", mission_count+1, "RUN", count, "\n------------\n")
            world_state = self.agent.getWorldState()
            current_state = self.get_state(world_state) # gets current state
            qvals = self.model.predict(current_state).numpy() # gets prediction of q values for current state
            print('got q values of current state:', qvals)

            # picks a random action or picks best action based on random probability
            if random.random() < self.epsilon: 
                action = np.random.choice(5, 1, p=self.action_weights)[0]
                print('choosing random action ', self.actions[action])
            else:
                action = np.argmax(qvals)
                print('choosing best action ', self.actions[action])

            perform_action_count = 1
            if self.explore and action!=4:
                perform_action_count = random.randint(1, 3)
                print('performing action %d times' % perform_action_count)
            action_count = 0
            self.agent.sendCommand(self.actions[action])
            time.sleep(1)
            world_state = self.agent.getWorldState()
            new_state = self.get_state(world_state)
            while action_count < perform_action_count and not np.all(current_state==new_state):
                reward = sum(r.getValue() for r in world_state.rewards) # fetches reward for the state
                print('reward at current state:', reward)
                self.experiences.append([current_state, action, reward, new_state]) # appends (s, a, r, s')
                print('saved the experience of the agent')
                current_state = new_state
                self.agent.sendCommand(self.actions[action])
                time.sleep(1)
                world_state = self.agent.getWorldState()
                new_state = self.get_state(world_state)
                action_count+=1
 
            # finds a random number of random (s, a, r, s') sets from d
            randindexes = random.sample(range(0, len(self.experiences)), random.randint(0, len(self.experiences)-1))
            print('random indexes chosen to update:', randindexes)
            index_count = 0
            no_of_experiences = len(randindexes)
            experiences_to_train = no_of_experiences
            while experiences_to_train>0:
                if experiences_to_train>self.batch_size:
                    indexes = randindexes[index_count : index_count + self.batch_size] 
                    index_count+=self.batch_size
                else:
                    indexes = randindexes[index_count:]
                    index_count+=no_of_experiences - index_count
                # self.update_q_network(indexes)
                self.double_update_q_network(indexes)
                experiences_to_train = no_of_experiences - index_count

            world_state = self.agent.getWorldState()
            count+=1
        if self.explore:
            self.model.save_weights(self.explore)
        
    def test(self):
        world_state = self.agent.getWorldState()
        total_rewards = 0

        while world_state.is_mission_running:
            current_state = self.get_state(world_state)
            
            # Fetch q-values for the current state
            qvals = self.model.predict(current_state).numpy()
            print('Got Q values of current state:', qvals)
            
            # Choose best action
            action = np.argmax(qvals)
            print('Choosing best action ', self.actions[action])

            # Perform best action
            self.agent.sendCommand(self.actions[action])
            time.sleep(1)
            world_state = self.agent.getWorldState() # Update world state

            current_reward = sum(r.getValue() for r in world_state.rewards) # Fetches reward for the state
            print('Reward at current state:', current_reward)
            total_rewards += current_reward
            print('Total rewards at this point:', total_rewards)
        
        return total_rewards