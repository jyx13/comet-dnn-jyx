# These scripts are based on the example of TensorFlow CIFAR-10
# They use the Apache lisence.  We will use the same.
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:

Usage:

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import shutil

import tensorflow as tf

import comet_dnn
import comet_dnn_input

FLAGS = tf.app.flags.FLAGS

# Output flags
tf.app.flags.DEFINE_string('model_dir', '/tmp/comet_dnn_model',
                           """Directory where to write train and test """
                           """directories""")
tf.app.flags.DEFINE_string('train_dir', None,
                           """Directory where to write summaries and """
                           """checkpoints.  Overrides the default """
                           """FLAGS.model_dir+'/train'""")
tf.app.flags.DEFINE_string('test_dir', None,
                           """Directory where to write testing summaries"""
                           """Overrides the default """
                           """FLAGS.model_dir+'/test'""")
tf.app.flags.DEFINE_string('saved_model_dir', None,
                           """Directory where to read SavedModel."""
                           """Overrides the default """
                           """FLAGS.model_dir+'/SavedModel'""")
tf.app.flags.DEFINE_integer('max_steps', None,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_dev_place', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1,
                            """How often to summerize the results.""")
tf.app.flags.DEFINE_integer('evaluation_frequency', None,
                            """How often to evaluate the test sample.""")
tf.app.flags.DEFINE_integer("num_checkpoints_steps", 10000,
                            """Number of steps per check point you want to """
                            """save.""")
tf.app.flags.DEFINE_boolean("debug_mode", False,
                            """Whether to active debug mode""")
tf.app.flags.DEFINE_boolean("save_image", False,
                            """Whether to save images""")




def _print_summary(batch_size, sec_per_batch, step, train_loss, test_loss):
    """
    Print a summary of the current run
    """
    format_str = '%s: step %d, train_loss = %.4f test_loss %.4f '+\
                 '(%.1f examples/sec; %.3f sec/batch)'
    print(format_str % (datetime.now(), step, train_loss, test_loss,
                        batch_size/sec_per_batch, sec_per_batch))

def train(training_files, testing_files):
    # Open a graph
    with tf.Graph().as_default():

        # Get or create the global step
        global_step = tf.train.get_or_create_global_step()

        # Compile which labels will be trained for
        lbls_to_train = [FLAGS.train_p_t, FLAGS.train_p_z,
                         FLAGS.train_entry_x, FLAGS.train_entry_y, FLAGS.train_entry_z,
                         FLAGS.train_vert_x, FLAGS.train_vert_y, FLAGS.train_vert_z,
                         FLAGS.train_n_turns ]
        print("Labels to train:",lbls_to_train)

        # Create placeholder images
        image_shape = [FLAGS.batch_size] + comet_dnn_input.IMAGE_SHAPE
        batch_images = tf.placeholder(tf.float32,
                                      shape=image_shape,
                                      name="input_images")
        # Create placeholder labels
        label_shape = [FLAGS.batch_size] + comet_dnn_input.LABEL_SHAPE
        batch_labels = tf.placeholder(tf.float32,
                                      shape=label_shape,
                                      name="input_labels")
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up
        # on GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            # Get the training dataset and a one-shot iterator
            train_data = comet_dnn_input.read_tfrecord_to_dataset(
                training_files,
                compression="GZIP",
                buffer_size=FLAGS.input_buffer_size,
                batch_size=FLAGS.batch_size,
                epochs=FLAGS.epochs,
                seed=FLAGS.random_seed)
            train_iter = train_data.make_one_shot_iterator()
            train_images, train_labels = train_iter.get_next()
            print(train_labels)

            # Get the testing dataset and reinitializable iterator
            test_data = comet_dnn_input.read_tfrecord_to_dataset(
                testing_files,
                compression="GZIP",
                buffer_size=FLAGS.input_buffer_size,
                batch_size=FLAGS.batch_size,
                epochs=1,
                seed=FLAGS.random_seed)
            test_iter = test_data.make_initializable_iterator()
            test_images, test_labels = test_iter.get_next()                    

        # Save images
        if (FLAGS.save_image==True):
            tf.summary.image("train_images_q", train_images[:,:,:,:1])
            tf.summary.image("train_images_t", train_images[:,:,:,1:])
            tf.summary.image("test_images_q", train_images[:,:,:,:1])
            tf.summary.image("test_images_t", train_images[:,:,:,1:])

        # Build a Graph that the regresses the values from the images
        predictions = comet_dnn.inference(batch_images)
        # Calculate loss using the labels
        loss = comet_dnn.loss(predictions, batch_labels)
        loss = tf.check_numerics(loss, message="NaN or Infinity loss")
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters
        train_op = comet_dnn.train(loss, global_step)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000) # don't cap the number of checkpoints to be made

        # Create a builder (to be used instead of train.Saver).
        if os.path.exists(FLAGS.model_dir):
            shutil.rmtree(FLAGS.model_dir)
            print("Removed extant model directory")
        os.makedirs(FLAGS.model_dir)
        print("Created new model directory")

        # Get the average total loss for the test epoch
        eval_loss = tf.placeholder(tf.float32)
        eval_n_batches = tf.placeholder(tf.float32)
        norm_eval_loss = tf.divide(eval_loss, eval_n_batches)
        eval_loss_summary = tf.summary.scalar("total_loss", norm_eval_loss)

        # Start running operations on the Graph.
        sess_config = tf.ConfigProto(log_device_placement=FLAGS.log_dev_place)
        with tf.Session(config=sess_config) as sess:
            # Define the summary writer
            train_summary_writer = \
                    tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            test_summary_writer = \
                    tf.summary.FileWriter(FLAGS.test_dir, sess.graph)
            # Initialize all the variables
            sess.run(init)
            step = -1
            # Run the training loop
            keep_running = True
            while keep_running:
                # Increment the step number
                step += 1
                if step == 1: # just for testing purposes, to interrupt training after some number of steps
                    keep_running = False
                try:
                    # Get the start time
                    start_time = time.time()
                    # Prepare information for this train_feed
                    evaluated_train_labels = train_labels.eval()
                    # Move columns of desired predictions to front of labels array
                    col_i = 0
                    for lbl_i in range(len(lbls_to_train)):
                        if lbls_to_train[lbl_i]:
                            evaluated_train_labels[:,col_i] = evaluated_train_labels[:,lbl_i]
                            col_i += 1

                    # Define final train_feed
                    train_feed = {batch_images : train_images.eval(),
                                  batch_labels : evaluated_train_labels}

                    # Run the session--this line does all the heavy computing
                    _, train_pred, train_loss = sess.run([train_op, predictions, loss],
                                                         feed_dict=train_feed)

                    graph = tf.get_default_graph()
                    #for op in graph.get_operations():
                    #    print(op.name)
                    print(graph.get_tensor_by_name("conv1/weights:0").eval())

                    # Debug train_feed, print only first element of labels
                    if(FLAGS.debug_mode==True):
                        print("----------------", "\n"
                              "Debug mode", "\n"
                              "train_labels", "\n"
                              "Type: ", type(train_feed[batch_labels]), "\n"
                              "Dim: " , train_feed[batch_labels].shape, "\n"
                              "First row label: " , train_feed[batch_labels][0], "\n",
                              "First row predi: " , train_pred[0], "\n",
                              "----------------", "\n")
                    sec_per_batch = time.time() - start_time

                    # Print some values to std_out
                    if step % FLAGS.summary_frequency == 0:
                        _print_summary(FLAGS.batch_size, sec_per_batch, step,
                                       train_loss, 999999.9)
                        summary_str = sess.run(summary_op, feed_dict=train_feed)
                        train_summary_writer.add_summary(summary_str, step)
                        # Add the loss comparison summary
                        train_summary_dict = {eval_loss : train_loss,
                                              eval_n_batches : 1}
                        train_summary = sess.run(eval_loss_summary,
                                                 feed_dict=train_summary_dict)
                        train_summary_writer.add_summary(train_summary, step)

                    # Save the model checkpoint periodically.
                    if step % FLAGS.num_checkpoints_steps == 0:
                        model_name = FLAGS.train_dir+'model-ckpt'
                        saver.save(sess, model_name, global_step=step) # save variables
                        builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.saved_model_dir+'step_'+str(step))
                        builder.add_meta_graph_and_variables(sess, ['test_tag'])
                        builder.save() # save full model
                    if FLAGS.max_steps is not None and step > FLAGS.max_steps:
                        keep_running = False

                    # Evaluate the testing sample
                    if (FLAGS.evaluation_frequency is not None) and \
                       (step % FLAGS.evaluation_frequency == 0):
                        # Initialize the iterator
                        sess.run(test_iter.initializer)
                        epoch_test_loss = 0.
                        test_step = 0
                        start_test_time = time.time()
                        keep_testing = True
                        while keep_testing:
                            try:
                                # Prepare information for this test_feed
                                evaluated_test_labels = test_labels.eval()
                                # Move columns of desired predictions to front of labels array
                                col_i = 0
                                for lbl_i in range(len(lbls_to_train)):
                                    if lbls_to_train[lbl_i]:
                                        evaluated_test_labels[:,col_i] = evaluated_test_labels[:,lbl_i]
                                        col_i += 1
                                # Define final TRAIN_feed
                                test_feed = {batch_images : test_images.eval(),
                                             batch_labels : evaluated_test_labels}
                                test_loss = sess.run(loss, feed_dict=test_feed)
                                epoch_test_loss += test_loss
                                test_step += 1
                            except tf.errors.OutOfRangeError:
                                total_test_time = time.time() - start_test_time
                                _print_summary(FLAGS.batch_size * test_step,
                                               total_test_time,
                                               test_step,
                                               888888.8,
                                               epoch_test_loss/test_step)
                                keep_testing = False

                        # Feed the summary dictionary the values
                        test_summary_dict = {eval_loss : epoch_test_loss,
                                             eval_n_batches : test_step}
                        test_summmary = sess.run(eval_loss_summary,
                                                 feed_dict=test_summary_dict)
                        test_summary_writer.add_summary(test_summmary, step)

                # Stop the training loop if we reach the end of the training loop
                except tf.errors.OutOfRangeError:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model-ckpt')
                    saver.save(sess, checkpoint_path, global_step=step) # save final variables
                    final_builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.saved_model_dir+'step_'+str(step))
                    final_builder.add_meta_graph_and_variables(sess, ["test_tag"], strip_default_attrs=False)
                    final_builder.save() # save full final model
                    keep_running = False


def main(argv=None):  # pylint: disable=unused-argument
    # Set the random seed
    FLAGS.random_seed = comet_dnn_input.set_global_seed(FLAGS.random_seed)
    # Dump the current settings to stdout
    comet_dnn.print_all_flags()

    # Check we have an input list
    if FLAGS.input_list is None:
        print("Error: No input file list given\n",
              "Please input a list of paths for tfrecord")
        # Return failure
        return 1

    # Read the input files and shuffle them
    training_files, testing_files = \
            comet_dnn_input.train_test_split_filenames(FLAGS.input_list,
                                                       FLAGS.percent_train,
                                                       FLAGS.percent_test,
                                                       FLAGS.random_seed)
    # Create a name for the train and test dirs
    if FLAGS.train_dir is None:
        FLAGS.train_dir = FLAGS.model_dir + "/train/"
    if FLAGS.test_dir is None:
        FLAGS.test_dir = FLAGS.model_dir + "/test/"
    if FLAGS.saved_model_dir is None:
        FLAGS.saved_model_dir = FLAGS.model_dir+"/SavedModel/"
    # Create check point directory
    if tf.gfile.Exists(FLAGS.train_dir):
        print("WARNING:", FLAGS.train_dir, "exists")
    else:
        for directory in [FLAGS.model_dir, FLAGS.train_dir, FLAGS.test_dir]:
            print(directory, " is made")
            tf.gfile.MakeDirs(directory)

    # Write the training and testing lists to a file
    comet_dnn_input.write_list_to_file(training_files,
                                       FLAGS.train_dir+"/training_files.txt")
    comet_dnn_input.write_list_to_file(testing_files,
                                       FLAGS.test_dir+"/testing_file.txt")
    # Train on the training set of files
    train(training_files, testing_files)
    # Return success
    return 0

if __name__ == '__main__':
    tf.app.run()

#    # Records the labels to the summary
#    with tf.variable_scope('lables') as scope:
#        for idx, a_name in enumerate(LABEL_NAMES_FULL):
#            tf.summary.histogram(a_name, labels[:,idx])
#
#    # Transform the image to color ima
#    with tf.variable_scope('images') as scope:
#        image_1_ch = image[:,:,:,:1]
#        tf.summary.image("hit_map", image_1_ch,
#                         max_outputs=tf.app.flags.FLAGS.max_output_images)
#    # Records the images to the summary
#    return image, labels
#
#        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
#        restored_step = 1
#        # Create a saver.
#        saver = tf.train.Saver(tf.global_variables())
#        # Define model path
#        if ckpt and ckpt.model_checkpoint_path:
#            print("======================================")
#            print("======================================\n")
#            print("Restore previous checkpoint")
#            saver.restore(sess, ckpt.model_checkpoint_path)
#            restored_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
#            print ("Restored step:",restored_step)
#        else:
#
