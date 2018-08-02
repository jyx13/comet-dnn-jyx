# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache license.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# CURRENT STATUS: eval_once() has been ignored in favor of evaluate().
# evaluate() is able to properly(?) restore a model from checkpoint (using either Saver or SavedModel),
# and then makes a few useful plots for model assessment.
# While restored variables are identical to the ones saved in checkpoints,
# histograms still look different from tensorboard by eye--reasons unclear.
# -- 2018/08/01, Jordan Xiao

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

def eval_plots(compiled_true_labels, compiled_predicted_labels, ckpt_num):
    # made this a separate function to keep evaluate() from being too cluttered
    # INPUTS: both have dimensions of batch_size rows by num_classes columns (ckpt_num for title purposes)
    # OUTPUTS: five plots per label
               
    # To improve:
    # 1. custom x-axis limits and binning for histograms (probably want to customize depending on which label)
    # 2. re-scale labels back into natural units (un-normalizing)

    # Compile which labels were trained for
    lbls_trained_for = [FLAGS.train_p_t, FLAGS.train_p_z,
                        FLAGS.train_entry_x, FLAGS.train_entry_y, FLAGS.train_entry_z,
                        FLAGS.train_vert_x, FLAGS.train_vert_y, FLAGS.train_vert_z,
                        FLAGS.train_n_turns ]
    print("Labels trained for:",lbls_trained_for)

    LABEL_NAMES = ["p_t", "p_z",
                   "entry_x", "entry_y", "entry_z",
                   "vert_x", "vert_y", "vert_z",
                   "n_turns"]

    print(LABEL_NAMES)
    # to avoid getting error in case lbls_trained_for is all 0s, re-order LABEL_NAMES instead of using new var
    # (same re-ordering technique as in train.py)
    i = 0
    for j in range(len(lbls_trained_for)):
        if lbls_trained_for[j]:
            LABEL_NAMES[i] = LABEL_NAMES[j]
            i += 1

    print(LABEL_NAMES)

    LABEL_LIMS = [] # to be filled in with suitable values for each label
    LABEL_NORMALIZE = [110.0, 155.,
                       110., 115., 125.,
                       22., 22., 85.,
                       5.] # remember to update this if the one in comet_dnn_inputs.py is changed

    n = 5 # number of plots; used to prevent figure overlapping, also makes it easier to add new plots

    # Get residuals (same dimensions as label arrays)
    compiled_residuals = compiled_true_labels - compiled_predicted_labels

    for i in np.arange(FLAGS.num_classes): # i is index for cycling through labels
        lbl_name = LABEL_NAMES[i]
        
        true_labels = compiled_true_labels[:,i]
        predicted_labels = compiled_predicted_labels[:,i]
        residuals = compiled_residuals[:,i]

        # Histograms
        plt.figure(i*n+1)
        plt.hist(true_labels,30) # bins=np.linspace(0.1,1.1,128))
        # plt.xlim([0.1,1.1])
        plt.grid(True)
        plt.xlabel("True values (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" true histogram (Ckpt "+ckpt_num+")")

        plt.figure(i*n+2)
        plt.hist(predicted_labels,30) # bins=np.linspace(-1.0,1.0,128))
        # plt.xlim([-1,1])
        plt.grid(True)
        plt.xlabel("Predicted values (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" predictions histogram (Ckpt "+ckpt_num+")")

        plt.figure(i*n+3)
        plt.hist(residuals,30)
        plt.xlabel("True - predicted (normalized)")
        plt.ylabel("Count")
        plt.title(lbl_name+" residuals histogram (Ckpt "+ckpt_num+")")

        buff = 0.1 # buffer for scatterplot axes limits; scaled to normalized data

        # Colored correlation scatterplot
        plt.figure(i*n+4)
        plt.scatter(true_labels, predicted_labels, c=abs(residuals))
        plt.plot([0,1],[0,1],'-k')
        plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
        plt.ylim(min(predicted_labels)-buff, max(predicted_labels)+buff)
        plt.xlabel("True labels")
        plt.ylabel("Predicted labels")
        plt.title(lbl_name+" true vs. predicted (Ckpt "+ckpt_num+")")

        # Residuals scatterplot
        plt.figure(i*n+5)
        plt.scatter(true_labels, residuals)
        plt.plot([0,1],[0,0],'-b')
        plt.xlim(min(true_labels)-buff, max(true_labels)+buff)
        plt.ylim(min(residuals)-buff, max(residuals)+buff)
        plt.xlabel("True labels")
        plt.ylabel("Residuals")
        plt.title(lbl_name+" true vs. residuals (Ckpt "+ckpt_num+")")

    plt.show(block = FLAGS.show_eval_plots) # block = True means show plots


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
#            tensor_to_print = "predictions/weights" # if want to print individual tensor from graph

#            print("Printing tensor(s) in checkpoint file:")
#            print_tensors_in_checkpoint_file(file_name=ckpt_name,
#                                             tensor_name=tensor_to_print,
#                                             all_tensors="",
#                                             all_tensor_names="")

#            print(np.shape(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
#            for tensor in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#                print(tensor.name)
#            print(graph.get_tensor_by_name(tensor_to_print+":0"))
#            print(graph.get_tensor_by_name(tensor_to_print+":0").eval())

#            for op in graph.get_operations():
#                print(op.name)
#                print(op)
#            print(graph.get_tensor_by_name("predictions/predictions:0"))

            # Get desired true labels
            all_true_labels = true_labels.eval() # turn true_labels tensor into array
            true_labels = all_true_labels[:,0:FLAGS.num_classes] # only take columns we bothered predicting for
            print("True labels:", true_labels)

            # Make feed_dict and run predictions operation
            print("Running predictions operation...")
            pred_feed = {batch_images: pred_images.eval()}
            predicted_labels = sess.run(predictions, feed_dict = pred_feed)
            print("Predictions:", predicted_labels)
 
            # Get some useful plots
            eval_plots(true_labels, predicted_labels, ckpt_num)

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

    # Figure out where saver and savedmodel checkpoint files, using same format defined in train.py
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
