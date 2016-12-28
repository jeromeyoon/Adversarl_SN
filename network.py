from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,batch_size,df_dim):
	self.batch_size = batch_size
	self.df_dim = df_dim
  
    def generator(self,nir):
	
	g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        g_nir0 =lrelu(g_bn0(conv2d(nir,self.df_dim*2,name='g_nir0')))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        g_nir1 =lrelu(g_bn1(conv2d(g_nir0,self.df_dim*4,name='g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        g_nir2 =lrelu(g_bn2(conv2d(g_nir1,self.df_dim*4,name='g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='g_bn4')
        g_nir3 =lrelu(g_bn3(conv2d(g_nir2,self.df_dim*2,name='g_nir3')))
        g_nir4 =conv2d(g_nir3,3,name='g_nir4')
	return tf.tanh(g_nir4),(g_nir4+1.)/2.0


    def discriminator(self, image,keep_prob, reuse=False):
	if reuse:
            tf.get_variable_scope().reuse_variables()   
	h0 = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2,name='high_d_h0_conv')) #output size: 128x128
	d_bn1 = batch_norm(self.batch_size,name='high_d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='high_d_h1_conv'))) #output size: 64x64
	d_bn2 = batch_norm(self.batch_size,name='high_d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='high_d_h2_conv'))) #output size: 32x32
	#d_bn3 = batch_norm(self.batch_size,name='high_d_bn3')
        h3 = conv2d(h2, 1,d_h=2,d_w=2,name='high_d_h3_conv') #output size: 16x16

        return tf.nn.sigmoid(h3)

