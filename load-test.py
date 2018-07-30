# restore Sam's model (trained july 4)
# TO DO: figure out how to give this model new data to make inferences

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/Jyx/Desktop/comet_stuff/comet_dnn/modules')
import comet_dnn
import comet_dnn_input
import comet_dnn_train
from read_tfrecord import read_tfrecord_to_array


print("COMET-DNN modules successfully imported")

meta_file = '/Users/Jyx/Desktop/comet_stuff/20180704/0001/00000001/train/model.ckpt-8919.meta' # FILENAME of meta file
ckpt_path = '/Users/Jyx/Desktop/comet_stuff/20180704/0001/00000001/train/' # DIRECTORY where checkpoint file is

filename = '/Users/Jyx/Desktop/comet_stuff/input/oa_xx_xxx_09010000-0000_xerynzb6emaf_user-TrkTree_000_500signal-label.tfrecord' # file to make inference on, one file = 256 examples by default
filenames = list(filename)

n_samps = 256
images = []
labels = []
compression = tf.python_io.TFRecordCompressionType.GZIP
tf_io_opts = tf.python_io.TFRecordOptions(compression)
tf_images, tf_labels = read_tfrecord_to_array(filename, tf_io_opts)
compiled_images = np.zeros((n_samps,18,300,2))

batch_images = tf.placeholder(tf.float32, shape = compiled_images.shape)
predictions = comet_dnn.inference(batch_images) # takes batch_images placeholder tensor as input

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    # restore already-trained model
    saver = tf.train.import_meta_graph(meta_file)
    print("Meta graph imported w/o errors")
    saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))
    print("Checkpoint restored w/0 errors")

    graph = tf.get_default_graph()
#    for op in graph.get_operations():
#        print(op.name)
    test_op = graph.get_tensor_by_name("conv1/weights/read:0")
#    print(test_op)
#    print(test_op.eval())

    # get 4-d array for batch_images in same way as plot_sim_tracks.py
    # create coordinator and run QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    # turn images into arrays and add to compilation arrays

    for samp_index in range(n_samps):
        print("Extracting sample",samp_index) # to separate samples
        img, lbls = sess.run([tf_images, tf_labels])
        images += [img]
        labels += [lbls]

    # stop threads                                                                                                                
    coord.request_stop()
    coord.join(threads)
    print("--------------")
    print("Threads closed")

#    print("Label 1:",labels[0])

    # stacking images into one array to give as feed_dict
    for img_index in range(len(images)):
        compiled_images[img_index,:,:,:] += images[img_index][0,:,:,:]
        
    # making feed_dict and running predictions
    pred_feed = {batch_images: compiled_images}
    output = sess.run(predictions,feed_dict = pred_feed)
    
#    output_summary = tf.summary.histogram("outputs",output)
#    sess.run(init_op)
#    sess.run(tf.summary.merge_all())

    print(output) # note: weights change every time unless random seed is set to constant value
    plt.hist(output,30)
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.title("Predictions histogram")
    plt.show(block = False) # change to true if want to see histogram

### NOTE: work moved to comet_dnn_eval.py, evaluate() function

#    pred_data = comet_dnn_input.read_tfrecord_to_dataset(filenames) # forget these for now (see comet_dnn_train.py, train method for reference)
#    pred_iter = pred_data.make_one_shot_iterator()
#    pred_images, pred_labels = pred_iter.get_next()
#    NEED TO INITIALIZE THE ITERATOR OBJECT--that's why it wasn't working previously
#    print("Images to be predicted: ", pred_images)
#    print("Label true values: ", pred_labels)
#    print("Labels evaluated: ", pred_labels.eval())
    



#    test_tensor_restore = graph.get_tensor_by_name("predictions/zero_fraction/Const:0")
#    print(test_tensor_restore.eval())
#    learning_rate = graph.get_tensor_by_name("learning_rate:0")
#    print("Learning rate: ",learning_rate.eval())
#    print(graph.get_tensor_by_name("conv1/weights/Initializer/truncated_normal/stddev:0").eval())

