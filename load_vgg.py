import tensorflow as tf
import numpy as np
import scipy.io, pdb
class VGG(object):
    def __init__(self,weights):
        data  = scipy.io.loadmat(weights)
	self.weights = np.squeeze(data['layers'])
		

    def vgg_net(self,image,reuse=False):
        layers = (
         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
         'relu5_3', 'conv5_4', 'relu5_4'
     )
        net = {}
        current = image
 	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):	
            for i, name in enumerate(layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = self.weights[i][0][0][0][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = self.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                    bias =  self.get_variable(bias.reshape(-1), name=name + "_b")
                    current = self.conv2d_basic(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current, name=name)
                elif kind == 'pool':
                    current = self.avg_pool_2x2(current)
                net[name] = current
	    current = self.avg_pool_2x2(net['conv5_3'])
	    net['pool5'] = current	
 
            return self.gramMatrix(net['pool2']), self.gramMatrix(net['pool5'])

    def conv2d_basic(self,x, W, bias):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(conv, bias)

    def avg_pool_2x2(self,x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    def get_variable(self,weights,name):
        init = tf.constant_initializer(weights, dtype=tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
        return var

    def gramMatrix(self,t):
	dims = t.get_shape().as_list()
	size = tf.size(t)
	t= tf.reshape(t,[-1,dims[3]])
	if dims[1] *dims[2] < dims[3]:

	   return tf.matmul(t,t,transpose_b=True)
	else:
	   return tf.matmul(t,t,transpose_a=True)
