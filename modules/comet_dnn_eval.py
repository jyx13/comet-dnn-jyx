# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache license.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# CURRENT STATUS: eval_once() has been ignored in favor of evaluate().
# evaluate() is able to properly(?) restore a model from checkpoint (using either Saver or SavedModel),
# and then makes a few useful plots for model assessment.
# While restored variables are identical to the ones saved in checkpoints,
# they still look different from tensorboard by eye--reasons unclear.
# -- 2018/07/31, Jordan Xiao

"""Evaluation for comet_dnn.

Accuracy:

Speed:

Usage:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import comet_dnn_input
import comet_dnn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', None,
                           """Directory where to write train and test """
                           """directories""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/comet_dnn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_boolean('eval_test', True,
                           """If true, evaluates the testing data""")
tf.app.flags.DEFINE_string('saver_ckpt_dir', None,
                           """Directory where to read checkpoints (for TF Saver).""")
tf.app.flags.DEFINE_string('saved_model_dir', None,
                           """Directory where to read saved model (for TF SavedModel).""")
tf.app.flags.DEFINE_string('checkpoint_num', None,
                           """Number of checkpoint to be reloaded.""")
tf.app.flags.DEFINE_boolean('show_eval_plots', True,
                           """If true, show histograms and correlation plots.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def eval_once(saver, summary_writer, batch_predictions, batch_labels, summary_op): # not currently in use
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    loss: From the prediction
    summary_op: Summary op.
    """
    # Get the loss of mean of batch images
    with tf.variable_scope("mean_loss_eval"):
        mean_loss = comet_dnn.loss(batch_predictions,batch_labels)

    # Get physics prediction
    predictions=tf.squeeze(batch_predictions)
    residual = predictions -  batch_labels[:,0]
    # Add summary
    tf.summary.histogram('/residual', residual)
    # define global number of images    
    eval_index = 0
    
    # It seems like by default it is getting the latest check point
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # Get path to the check point
    path_to_ckpt = ckpt.model_checkpoint_path

    global_step = path_to_ckpt.split('/')[-1].split('-')[-1]
    eval_index = int(global_step)
    
    with tf.Session() as sess:
        # Check if we have the checkpoint and path exist
        if ckpt and path_to_ckpt:
            # Restores from checkpoint
            saver.restore(sess, path_to_ckpt)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/comet_dnn_train/model.ckpt-0,
            # extract global_step from it.

            print_tensors_in_checkpoint_file(file_name=path_to_ckpt,
                                             tensor_name="",
                                             all_tensors="",
                                             all_tensor_names="") 
            # Open summary
            print(eval_index)
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            # Add value to summary
            loss=sess.run(mean_loss)
            summary.value.add(tag='pt_residual @ 1', simple_value=loss)
            #print("%d mean_loss %f " % (eval_index,loss))
            # Add summary 
            summary_writer.add_summary(summary,eval_index)
            eval_index = eval_index + 1
        else:
            print('No checkpoint file found')
            return


def evaluate(eval_files):

    ckpt_num = FLAGS.checkpoint_num # string (for filepath concat)

    # for Saver
    ckpt_name = FLAGS.saver_ckpt_dir + 'model-ckpt-' + ckpt_num # full path to checkpoint PREFIX (no file extensions)  
    meta_file = ckpt_name + '.meta' # full filepath to meta file

    # for SavedModel
    model_dir = FLAGS.saved_model_dir + 'step_' + ckpt_num # full path to directory where SavedModel is stored

    with tf.Graph().as_default():
        # Extracting data
        pred_data = comet_dnn_input.read_tfrecord_to_dataset(
            eval_files,
            compression="GZIP",
            buffer_size=2e9,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            seed=FLAGS.random_seed)
        pred_iter = pred_data.make_one_shot_iterator()
        pred_images, true_labels = pred_iter.get_next()
        
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        
        with tf.Session() as sess:
            print("Entering session...")
            # initialize variables
            sess.run(init_op)

            # Restore model, using Saver
#            print("Restoring checkpoint:" + ckpt_name)
#            saver = tf.train.import_meta_graph(meta_file)
#            saver.restore(sess, ckpt_name)

            # Restore model, using SavedModel
            print("Reloaded model path:"+model_dir)
            tf.saved_model.loader.load(sess, ["test_tag"], model_dir)            

            # Reloading predictions operation and corresponding placeholder variable
            graph = tf.get_default_graph()
            predictions = graph.get_tensor_by_name("predictions/predictions:0")
            batch_images = graph.get_tensor_by_name("input_images:0")

            # A BUNCH OF PRINT STATEMENTS FOR DEBUGGING MODEL RESTORE
            tensor_to_print = "predictions/weights" # if want to print individual tensor from graph

            print("Printing tensor(s) in checkpoint file:")
            print_tensors_in_checkpoint_file(file_name=ckpt_name,
                                             tensor_name=tensor_to_print,
                                             all_tensors="",
                                             all_tensor_names="")

#            print(np.shape(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
#            for tensor in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#                print(tensor.name)
            print(graph.get_tensor_by_name(tensor_to_print+":0"))
            print(graph.get_tensor_by_name(tensor_to_print+":0").eval())

#            for op in graph.get_operations():
#                print(op.name)
#                print(op)
#            print(graph.get_tensor_by_name("predictions/predictions:0"))

            # Make feed_dict and run predictions operation
            print("Running predictions operation...")
            pred_feed = {batch_images: pred_images.eval()}
            output = sess.run(predictions,feed_dict = pred_feed)
            predicted_labels = output[:,FLAGS.num_classes-1]

            print("Predictions:",predicted_labels)
#            print("Predictions size:",predicted_labels.shape)
            
            all_true_labels = true_labels.eval() # turn true_labels tensor into array
            true_labels = all_true_labels[:,FLAGS.num_classes-1] # only take columns we bothered predicting for
            print("True labels:", true_labels)
#            print(true_labels.shape)
   
            residuals = true_labels - predicted_labels
#            print(residuals.shape)
            
            # Histograms
            plt.figure(1)
            plt.hist(true_labels,bins=np.linspace(0.1,1.1,128))
            plt.xlim([0.1,1.1])
            plt.grid(True)
            plt.xlabel("True values (normalized)")
            plt.ylabel("Count")
            plt.title("True values histogram (Ckpt "+ckpt_num+")")
            
            plt.figure(2)
            plt.hist(predicted_labels,bins=np.linspace(-1.0,1.0,128))
            plt.xlim([-1,1])
            plt.grid(True)
            plt.xlabel("Predicted values (normalized)")
            plt.ylabel("Count")
            plt.title("Predictions histogram (Ckpt "+ckpt_num+")")
    
            plt.figure(3)
            plt.hist(residuals,30)
            plt.xlabel("True - predicted (normalized)")
            plt.ylabel("Count")
            plt.title("Residuals histogram (Ckpt "+ckpt_num+")")
    
            buff = 0.1 # buffer for axes limits
    
            # Colored correlation scatterplot
            plt.figure(4)
            plt.scatter(true_labels, predicted_labels, c=abs(residuals))
            plt.plot([0,1],[0,1],'-k')
            plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
            plt.ylim(min(predicted_labels)-buff, max(predicted_labels)+buff)
            plt.xlabel("True labels")
            plt.ylabel("Predicted labels")
            plt.title("True vs. predicted (Ckpt "+ckpt_num+")")

            # Residuals scatterplot
            plt.figure(5)
            plt.scatter(true_labels, residuals)
            plt.plot([0,1],[0,0],'-b')
            plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
            plt.ylim(min(residuals)-buff, max(residuals)+buff)
            plt.xlabel("True labels")
            plt.ylabel("Residuals")
            plt.title("True vs. residuals (Ckpt "+ckpt_num+")")
    
            plt.show(block = FLAGS.show_eval_plots) # change to true if want to see plots

"""
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.move_avg_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
        
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    while True:
        eval_once(saver, summary_writer, predictions, batch_labels, summary_op)
        if FLAGS.run_once:
            break
        time.sleep(1)
"""

def main(argv=None):  # pylint: disable=unused-argument

    # Set the random seed
    FLAGS.random_seed = comet_dnn_input.set_global_seed(FLAGS.random_seed)
    # Dump the current settings to stdout
    print("FLAGS:----------------------------")
    comet_dnn.print_all_flags()
    print("----------------------------------")
    # Read the input files and shuffle them
    # TODO read these from file list found in train_dir
    training_files, testing_files = \
            comet_dnn_input.train_test_split_filenames(FLAGS.input_list,
                                                       FLAGS.percent_train,
                                                       FLAGS.percent_test,
                                                       FLAGS.random_seed)

    # Create a name for the train and test dirs
    if FLAGS.saver_ckpt_dir is None:
        FLAGS.saver_ckpt_dir = FLAGS.model_dir + "/train/"
    if FLAGS.saved_model_dir is None:
        FLAGS.saved_model_dir = FLAGS.model_dir+"/SavedModel/"

    # Evaluate the testing files by default
    eval_files = testing_files
    if not FLAGS.eval_test:
        eval_files = training_files
    # Reset the output directory
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    # Evaluate the files
    evaluate(eval_files)

if __name__ == '__main__':
    tf.app.run()
