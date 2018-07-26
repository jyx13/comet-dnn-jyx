# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache lisence.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
# TODO get this file working

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
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import comet_dnn_input
import comet_dnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/comet_dnn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_boolean('eval_test', True,
                           """If true, evaluates the testing data""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/comet_dnn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def eval_once(saver, summary_writer, batch_predictions, batch_labels, summary_op):
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
    """Eval comet_dnn for a number of steps."""
    #with tf.Graph().as_default() as grph:
    # Get batch_images and batch_labels for comet_dnn.
    tf.reset_default_graph()

    batch_images, batch_labels = \
        comet_dnn_input.read_tfrecord_to_tensor(
        eval_files,
        compression="GZIP",
        buffer_size=2e9,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs)
    
    # Get the predictions
    predictions = comet_dnn.inference(batch_images)
    # Get the loss
    loss = comet_dnn.loss(predictions, batch_labels)
    
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


def main(argv=None):  # pylint: disable=unused-argument
    # Set the random seed
    FLAGS.random_seed = comet_dnn_input.set_global_seed(FLAGS.random_seed)
    # Dump the current settings to stdout
    comet_dnn.print_all_flags()
    # Read the input files and shuffle them
    # TODO read these from file list found in train_dir
    training_files, testing_files = \
            comet_dnn_input.train_test_split_filenames(FLAGS.input_list,
                                                       FLAGS.percent_train,
                                                       FLAGS.percent_test,
                                                       FLAGS.random_seed)
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
