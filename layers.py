import os
import tensorflow as tf
import numpy as np
from PIL import Image

tf.compat.v1.enable_eager_execution()

class Layers:
    def __init__(self, num_classes, learning_rate, save_model_name='weights.npy', weights_file=None):
        self.initializer = tf.initializers.glorot_uniform()
        self.num_classes = num_classes
        # Set adam optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.model_name = 'A'
        self.save_model_name = save_model_name

        if weights_file:
            weight_info = np.load(os.path.join('weights', weights_file), allow_pickle=True)
            # If best loss value in the numpy array, save it as a class variable along with weights
            # Else just save the weights
            if len(weight_info) == 13:
                self.weights, self.best_loss = weight_info[:12], weight_info[12]
            else:
                self.weights = weight_info
                self.best_loss = float('inf')
        else:
            # set weights initializer
            initializer = tf.compat.v1.keras.initializers.glorot_uniform()
            # Shapes of weight arrays of each layer
            shapes = [
                [7,7,3,64] , 
                [1,1,64,128] ,
                [3,3,128,32] , 
                [1,1,96,128] ,
                [3,3,128,32] , 
                [1,1,128,128] ,
                [3,3,128,32] , 
                [1,1,160,128] ,
                [3,3,128,32] , 
                [1,1,192,32] ,
                [93312 ,1024] ,
                [1024, self.num_classes] ,
            ]
            self.weights = []
            for i in range( len( shapes ) ):
                self.weights.append(self.get_weight(initializer, shapes[i], 'weight{}'.format(i)))
            self.best_loss = float('inf')

    # returns the weights after initializing with random distribution
    def get_weight( self, initializer, shape , name ):
        return tf.Variable(initializer(shape),name=name,trainable=True,dtype=tf.float32)

    # Save weights at model name
    def save_weights(self, explore):
        if 'weights' not in os.listdir('.'):
            os.mkdir('weights')

        if not explore and self.update_loss():
            weight_info = self.weights + [self.best_loss]
            np.save(os.path.join('weights',self.save_model_name), weight_info)
        else:
            weight_info = self.weights
            np.save(os.path.join('weights',self.save_model_name), weight_info)
    
    # If current loss is better than best loss, update
    def update_loss(self):
        if self.current_loss < self.best_loss:
            print('loss improved by %f, saving weights into %s' % (self.best_loss - self.current_loss, self.save_model_name))
            self.best_loss = self.current_loss
            return True
        return False

    # Return convolution layer 
    def conv2d(self, inputs, filters, strides, padding='VALID'):
        output = tf.nn.conv2d(inputs, filters, [1, strides, strides, 1], padding=padding)
        return tf.nn.leaky_relu(output, alpha=1)

    # Return MaxPool2D layer
    def maxpool(self, inputs, pool_size, strides, padding='VALID'):
        return tf.nn.max_pool2d(inputs, ksize=pool_size, padding=padding, strides=[1, strides, strides, 1])

    # Return AveragePool2D layer
    def avgpool(self, inputs, pool_size, strides, padding='VALID'):
        return tf.nn.avg_pool2d(inputs, ksize=pool_size, padding=padding, strides=[1, strides, strides, 1])
    
    # Return dense layer
    def dense(self, inputs, weights, dropout_rate):
        x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=1)
        if dropout_rate!=0:
            return tf.nn.dropout(x, rate=dropout_rate)
        return x

    def predict(self, x):
        # This model draws several logical concepts of Densenet Architecture

        # Input tensor of size 224*224
        input_tensor = tf.cast(x, dtype=tf.float32)
        # Initial Convolution layer of filter size 7*7 and strides 2
        initial_conv = self.conv2d(input_tensor, self.weights[0], strides=2)
        # Max Pooling layer of filter size 2
        max_pooling_initial = self.maxpool(initial_conv, pool_size=2, strides=1)

        # Activation layer
        batch_1_activ = tf.nn.relu(max_pooling_initial)
        # Convolution layer of k*4 number of filters with filter size (1,1)
        batch_1_conv2d_1 = self.conv2d(batch_1_activ, self.weights[1], strides=1, padding='SAME')
        # Dropout Layer
        batch_1_drop = tf.nn.dropout(batch_1_conv2d_1, rate=0.4)
        # Convolution Layer of k number of filters with filter size (3,3)
        batch_1_conv2d_2 = self.conv2d(batch_1_drop, self.weights[2], strides=1, padding='SAME')

        # Concatenate the the first and second block
        batch_2 = tf.concat([max_pooling_initial, batch_1_conv2d_2], axis=3)
        
        # Activation layer
        batch_2_activ = tf.nn.relu(batch_2)
        # Convolution layer
        batch_2_conv2d_1 = self.conv2d(batch_2_activ, self.weights[3], strides=1, padding='SAME')
        # Dropout Layer
        batch_2_drop = tf.nn.dropout(batch_2_conv2d_1, rate=0.4)
        # Convolution Layer
        batch_2_conv2d_2 = self.conv2d(batch_2_drop, self.weights[4], strides=1, padding='SAME')

        # Concatenate the the first and second block
        batch_3 = tf.concat([batch_2, batch_2_conv2d_2], axis=3)
        # print('batch_3', batch_3.shape)

        # Activation layer
        batch_3_activ = tf.nn.relu(batch_3)
        # Convolution layer
        batch_3_conv2d_1 = self.conv2d(batch_3_activ, self.weights[5], strides=1, padding='SAME')
        # Dropout Layer
        batch_3_drop = tf.nn.dropout(batch_3_conv2d_1, rate=0.4)
        # Convolution Layer
        batch_3_conv2d_2 = self.conv2d(batch_3_drop, self.weights[6], strides=1, padding='SAME')

        # Concatenate the the first and second block
        batch_4 = tf.concat([batch_3, batch_3_conv2d_2], axis=3)

        # Activation layer
        batch_4_activ = tf.nn.relu(batch_4)
        # Convolution layer
        batch_4_conv2d_1 = self.conv2d(batch_4_activ, self.weights[7], strides=1, padding='SAME')
        # Dropout Layer
        batch_4_drop = tf.nn.dropout(batch_4_conv2d_1, rate=0.4)
        # Convolution Layer
        batch_4_conv2d_2 = self.conv2d(batch_4_drop, self.weights[8], strides=1, padding='SAME')

        # Concatenate the the first and second block
        final_batch = tf.concat([batch_4, batch_4_conv2d_2], axis=3)

        # Downsampling Activation Layer
        downsampling_activ = tf.nn.relu(final_batch)
        # Downsampling Convolution Layer
        downsampling_conv2d_1 = self.conv2d(downsampling_activ, self.weights[9], strides=1, padding='VALID')
        # Average Pooling Layer
        downsampling_average = self.avgpool(downsampling_conv2d_1, pool_size=2, strides=2)

        # Flatten Layer
        flatten = tf.reshape(downsampling_average, shape=(tf.shape(downsampling_average)[0], -1))
        # Dense Layer of 1024 units
        top_layer_dense_1 = self.dense(flatten, self.weights[10], dropout_rate=0.5)
        # Dense Layer of num_classes units
        top_layer_dense_2 = self.dense(top_layer_dense_1, self.weights[11], dropout_rate=0)
        # Return Softmax value
        return top_layer_dense_2
    
    # Returns the loss of the current training step
    def loss(self, pred , target, regularization_parameter):
        # Sum l2 loss value of parameter for regularization
        regularizer = tf.nn.l2_loss(self.weights[0])
        for weight_index in range(1, len(self.weights)):
            regularizer += tf.nn.l2_loss(self.weights[weight_index])
        # Calculate MSE loss between predicted and expected output
        mse = tf.compat.v1.losses.mean_squared_error( target , pred )
        # Return MSE + regularization_parameter * regularizer_sum
        return tf.reduce_mean( mse + regularizer * regularization_parameter)

    # Updates the weights of the network using Adam Gradient Descent
    def train_step(self, inputs, outputs ):
        self.weights = list(self.weights)
        with tf.GradientTape() as tape:
            # Calculate loss
            current_loss = self.loss( self.predict( inputs ), outputs, 1)
            # compute gradient of the weights with the current loss
            grads = tape.gradient( target=current_loss , sources=self.weights )
            # Apply the gradients to the weights using the Adam optimizer
            self.optimizer.apply_gradients( zip( grads , self.weights ) )
            self.current_loss = current_loss.numpy()
            print('current loss: ', self.current_loss )
        return current_loss.numpy()

if __name__ == "__main__":
    model = Layers(num_classes= 10, learning_rate=0.01)
    # print(model.weights[0])
    image = Image.open('reference/test.png')
    np_image = np.asarray(image)
    np_image = np_image[np.newaxis, :, :, :]
    np_image = np_image/255.0
    output = model.predict(np_image)
    # print(type(output))
    print(output.numpy())
    print(np.argmax(output))
    # np.save('reference/test.npy', model.weights)
    # b = np.load('test.npy', allow_pickle=True)
    # print(len(b), b.shape)
    # print(b[0].shape)