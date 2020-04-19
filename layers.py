import os
import tensorflow as tf
import numpy as np
from PIL import Image

tf.compat.v1.enable_eager_execution()

class Layers:
    def __init__(self, num_classes, batch_size, learning_rate, save_model_name='weights.npy', weights_file=None):
        self.batch_size = batch_size
        self.initializer = tf.initializers.glorot_uniform()
        self.num_classes = num_classes
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.save_model_name = save_model_name

        if weights_file:
            self.weights = np.load(weights_file, allow_pickle=True)
        else:
            initializer = tf.compat.v1.keras.initializers.glorot_uniform()
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

    def get_weight( self, initializer, shape , name ):
        return tf.Variable(initializer(shape),name=name,trainable=True,dtype=tf.float32)

    def save_weights(self):
        if 'weights' not in os.listdir('.'):
            os.mkdir('weights')
        print('Saving weights into', self.save_model_name)
        np.save(os.path.join('weights',self.save_model_name), self.weights)

    def conv2d(self, inputs, filters, strides, padding='VALID'):
        output = tf.nn.conv2d(inputs, filters, [1, strides, strides, 1], padding=padding)
        return tf.nn.leaky_relu(output, alpha=1)

    def maxpool(self, inputs, pool_size, strides, padding='VALID'):
        return tf.nn.max_pool2d(inputs, ksize=pool_size, padding=padding, strides=[1, strides, strides, 1])

    def avgpool(self, inputs, pool_size, strides, padding='VALID'):
        return tf.nn.avg_pool2d(inputs, ksize=pool_size, padding=padding, strides=[1, strides, strides, 1])
    
    def dense(self, inputs, weights, dropout_rate):
        x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=1)
        if dropout_rate!=0:
            return tf.nn.dropout(x, rate=dropout_rate)
        return x

    def return_weight(self, shape, name=None):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    def predict(self, x):
        # This model draws several logical concepts of Densenet Architecture

        # Input tensor of size 224*224
        input_tensor = tf.cast(x, dtype=tf.float32)
        # Initial Convolution layer of filter size 7*7 and strides 2
        initial_conv = self.conv2d(input_tensor, self.weights[0], strides=2)
        # # print('initial_conv', initial_conv.shape)
        # Max Pooling layer of filter size 2
        max_pooling_initial = self.maxpool(initial_conv, pool_size=2, strides=1)
        # print('max_pooling_initial:', max_pooling_initial.shape)

        # Batch normalization layer
        # batch_1_batchNorm = tf.nn.batch_normalization(max_pooling_initial)
        # Activation layer
        batch_1_activ = tf.nn.relu(max_pooling_initial)
        # print('batch_1_activ', batch_1_activ.shape)
        # Convolution layer of k*4 number of filters with filter size (1,1)
        batch_1_conv2d_1 = self.conv2d(batch_1_activ, self.weights[1], strides=1, padding='SAME')
        # print('batch_1_conv2d_1', batch_1_conv2d_1.shape)
        # Dropout Layer
        batch_1_drop = tf.nn.dropout(batch_1_conv2d_1, rate=0.4)
        # print('batch_1_drop', batch_1_drop.shape)
        # Convolution Layer of k number of filters with filter size (3,3)
        batch_1_conv2d_2 = self.conv2d(batch_1_drop, self.weights[2], strides=1, padding='SAME')
        # print('batch_1_conv2d_2', batch_1_conv2d_2.shape)

        # Concatenate the the first and second block
        batch_2 = tf.concat([max_pooling_initial, batch_1_conv2d_2], axis=3)
        # print('batch_2', batch_2.shape)
        
        # Batch normalization layer
        # batch_2_batchNorm = tf.nn.batch_normalization(batch_2)
        # Activation layer
        batch_2_activ = tf.nn.relu(batch_2)
        # print('batch_2_activ', batch_2_activ.shape)
        # Convolution layer
        batch_2_conv2d_1 = self.conv2d(batch_2_activ, self.weights[3], strides=1, padding='SAME')
        # print('batch_2_conv2d_1', batch_2_conv2d_1.shape)
        # Dropout Layer
        batch_2_drop = tf.nn.dropout(batch_2_conv2d_1, rate=0.4)
        # print('batch_2_drop', batch_2_drop.shape)
        # Convolution Layer
        batch_2_conv2d_2 = self.conv2d(batch_2_drop, self.weights[4], strides=1, padding='SAME')
        # print('batch_2_conv2d_2', batch_2_conv2d_2.shape)

        # Concatenate the the first and second block
        batch_3 = tf.concat([batch_2, batch_2_conv2d_2], axis=3)
        # print('batch_3', batch_3.shape)

        # Batch normalization layer
        # batch_3_batchNorm = tf.nn.batch_normalization(batch_3)
        # Activation layer
        batch_3_activ = tf.nn.relu(batch_3)
        # print('batch_3_activ', batch_3_activ.shape)
        # Convolution layer
        batch_3_conv2d_1 = self.conv2d(batch_3_activ, self.weights[5], strides=1, padding='SAME')
        # print('batch_3_conv2d_1', batch_3_conv2d_1.shape)
        # Dropout Layer
        batch_3_drop = tf.nn.dropout(batch_3_conv2d_1, rate=0.4)
        # print('batch_3_drop', batch_3_drop.shape)
        # Convolution Layer
        batch_3_conv2d_2 = self.conv2d(batch_3_drop, self.weights[6], strides=1, padding='SAME')
        # print('batch_3_conv2d_2', batch_3_conv2d_2.shape)

        # Concatenate the the first and second block
        batch_4 = tf.concat([batch_3, batch_3_conv2d_2], axis=3)
        # print('batch_4', batch_4.shape)

        # Batch normalization layer
        # batch_4_batchNorm = tf.nn.batch_normalization(batch_4)
        # Activation layer
        batch_4_activ = tf.nn.relu(batch_4)
        # print('batch_4_activ', batch_4_activ.shape)
        # Convolution layer
        batch_4_conv2d_1 = self.conv2d(batch_4_activ, self.weights[7], strides=1, padding='SAME')
        # print('batch_4_conv2d_1', batch_4_conv2d_1.shape)
        # Dropout Layer
        batch_4_drop = tf.nn.dropout(batch_4_conv2d_1, rate=0.4)
        # print('batch_4_drop', batch_4_drop.shape)
        # Convolution Layer
        batch_4_conv2d_2 = self.conv2d(batch_4_drop, self.weights[8], strides=1, padding='SAME')
        # print('batch_4_conv2d_2', batch_4_conv2d_2.shape)

        # Concatenate the the first and second block
        final_batch = tf.concat([batch_4, batch_4_conv2d_2], axis=3)
        # print('final_batch', final_batch.shape)

        # Downsampling BatchNormalization
        # downsampling_batchNorm = tf.nn.batch_normalization(final_batch)
        # Downsampling Activation Layer
        downsampling_activ = tf.nn.relu(final_batch)
        # print('downsampling_activ', downsampling_activ.shape)
        # Downsampling Convolution Layer
        downsampling_conv2d_1 = self.conv2d(downsampling_activ, self.weights[9], strides=1, padding='VALID')
        # print('downsampling_conv2d_1', downsampling_conv2d_1.shape)
        # Average Pooling Layer
        downsampling_average = self.avgpool(downsampling_conv2d_1, pool_size=2, strides=2)
        # print('downsampling_average', downsampling_average.shape)

        # Flatten Layer
        flatten = tf.reshape(downsampling_average, shape=(tf.shape(downsampling_average)[0], -1))
        # print('flatten', flatten.shape)
        # Dense Layer of 1024 units
        top_layer_dense_1 = self.dense(flatten, self.weights[10], dropout_rate=0.4)
        # print('top_layer_dense_1', top_layer_dense_1.shape)
        # Dense Layer of num_classes units
        top_layer_dense_2 = self.dense(top_layer_dense_1, self.weights[11], dropout_rate=0)
        # print('top_layer_dense_2', top_layer_dense_2.shape)
        # Return Softmax value
        # print(type(top_layer_dense_2))
        return top_layer_dense_2
    
    def loss(self, pred , target, regularization_parameter):
        regularizer = tf.nn.l2_loss(self.weights[0])
        for weight_index in range(1, len(self.weights)):
            regularizer += tf.nn.l2_loss(self.weights[weight_index])
        mse = tf.compat.v1.losses.mean_squared_error( target , pred )
        return tf.reduce_mean( mse + regularizer * regularization_parameter)

    def train_step(self, inputs, outputs ):
        with tf.GradientTape() as tape:
            current_loss = self.loss( self.predict( inputs ), outputs, 100)
            grads = tape.gradient( target=current_loss , sources=self.weights )
            self.optimizer.apply_gradients( zip( grads , self.weights ) )
            print('Current loss: ', current_loss.numpy() )

if __name__ == "__main__":
    model = Layers(num_classes= 10, batch_size=1, learning_rate=0.01)
    # print(model.weights[0])
    image = Image.open('reference/test.png')
    np_image = np.asarray(image)
    np_image = np_image[np.newaxis, :, :, :]
    np_image = np_image/255.0
    output = model.predict(np_image)
    # print(type(output))
    print(output.numpy())
    print(np.argmax(output))
    # np.save('test.npy', model.weights)
    # b = np.load('test.npy', allow_pickle=True)
    # print(len(b), b.shape)
    # print(b[0].shape)