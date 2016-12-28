import numpy as np
import os
import tensorflow as tf
import random,time,json,pdb,scipy.misc,glob
from model_queue import DCGAN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
from numpy import inf
from sorting import natsorted
import matplotlib.image as mpimg
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "1228_Adversal_vgg", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    if not os.path.exists(os.path.join('./logs',time.strftime('%d%m'))):
    	os.makedirs(os.path.join('./logs',time.strftime('%d%m')))

    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,\
	    dataset_name=FLAGS.dataset,is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
	    dcgan = EVAL(sess, batch_size=1,ir_image_shape=[256,256,1],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
	    print('deep model test \n')

        if FLAGS.is_train:
	    pdb.set_trace()
            dcgan.train(FLAGS)
        else:
            list_val = [11,16,21,22,33,36,38,53,59,92]
	    print '1: Estimating Normal maps from arbitary obejcts \n'
	    print '2: Estimating Normal maps according to Light directions and object tilt angles \n'
	    x = input('Selecting a Evaluation mode:')
            VAL_OPTION = int(x)
            if VAL_OPTION ==1: # arbitary dataset 
                print("Computing arbitary dataset ")
		trained_models = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		trained_models  = natsorted(trained_models)
		datapath = '/research2/Ammonight/*.bmp'
                savepath = '/research2/Ammonight/output'
		mean_nir = -0.3313
		fulldatapath = os.path.join(glob.glob(datapath))
		model = trained_models[4]
		model = model.split('/')
		model = model[-1]
		dcgan.load(FLAGS.checkpoint_dir,model)
                for idx in xrange(len(fulldatapath)):
		    input_= scipy.misc.imread(fulldatapath[idx]).astype(float)
	            input_ = scipy.misc.imresize(input_,[600,800])
	            input_  = (input_/127.5)-1. # normalize -1 ~1
                    input_ = np.reshape(input_,(1,input_.shape[0],input_.shape[1],1)) 
                    input_ = np.array(input_).astype(np.float32)
		    mask = [input_>-1.0][0]*1.0
		    mean_mask = mask * mean_nir
		    #input_ = input_ - mean_mask
                    start_time = time.time() 
                    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
                    print('time: %.8f' %(time.time()-start_time))     
                    # normalization #
                    sample = np.squeeze(sample).astype(np.float32)
	            output = np.sqrt(np.sum(np.power(sample,2),axis=2))
		    output = np.expand_dims(output,axis=-1)
		    output = sample/output
		    output[output ==inf] = 0.0
		    sample = (output+1.0)/2.0

                    name = fulldatapath[idx].split('/')
		    name = name[-1].split('.')
                    name = name[0]
		    savename = savepath + '/normal_' + name +'.bmp' 
                    scipy.misc.imsave(savename, sample)

	    elif VAL_OPTION ==2: # depends on light sources 
                list_val = [11,16,21,22,33,36,38,53,59,92]
		mean_nir = -0.3313 #-1~1
		save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		save_files  = natsorted(save_files)
		savepath ='./Deconv_L1_result'
		if not os.path.exists(os.path.join(savepath)):
		    os.makedirs(os.path.join(savepath))
		selec_model=[-2]
		#[selec_model.append(ii) for ii in range(0,len(save_files),2)]
                for m in range(len(selec_model)):
		    model = save_files[selec_model[m]]
		    model = model.split('/')
		    model = model[-1]
		    dcgan.load(FLAGS.checkpoint_dir,model)
	            for idx in range(len(list_val)):
		        if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		            os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		        for idx2 in range(1,10): #tilt angles 1~9 
		            for idx3 in range(5,7): # light source 
			        print("Selected material %03d/%d" % (list_val[idx],idx2))
			        img = '/research2/ECCV_dataset_resized/save%03d/%d' % (list_val[idx],idx2)
				noise = np.random.rand(1,256,256,1)
				#noise = np.random.uniform(-1,1,size=(1,600,800,1))
			        input_ = scipy.misc.imread(img+'/%d.bmp' %idx3).astype(float) #input NIR image
			        input_ = scipy.misc.imresize(input_,[256,256])
			        input_  = input_/127.5 -1.0 # normalize -1 ~1
			        input_ = np.reshape(input_,(1,256,256,1)) 
			        input_ = np.array(input_).astype(np.float32)
			        gt_ = scipy.misc.imread(img+'/12_Normal.bmp').astype(float)
			        gt_ = np.sum(gt_,axis=2)
			        gt_ = scipy.misc.imresize(gt_,[256,256])
			        gt_ = np.reshape(gt_,[1,256,256,1])
			        mask =[gt_ >0.0][0]*1.0
			        mean_mask = mean_nir * mask
			        #input_ = input_ - mean_mask	
			        start_time = time.time() 
			        sample  = sess.run(dcgan.G, feed_dict={dcgan.ir_images: input_})
			        #sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
			        print('time: %.8f' %(time.time()-start_time))     
			        # normalization #
			        sample = np.squeeze(sample).astype(np.float32)
			        output = np.sqrt(np.sum(np.power(sample,2),axis=2))
			        output = np.expand_dims(output,axis=-1)
			        output = sample/output
			        output = (output+1.)/2.
			        if not os.path.exists(os.path.join(savepath,'%03d/%d/%s' %(list_val[idx],idx2,model))):
			            os.makedirs(os.path.join(savepath,'%03d/%d/%s' %(list_val[idx],idx2,model)))
			        savename = os.path.join(savepath,'%03d/%d/%s/single_normal_%03d.bmp' % (list_val[idx],idx2,model,idx3))
			        scipy.misc.imsave(savename, output)


if __name__ == '__main__':
    tf.app.run()
