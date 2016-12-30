import os
import math
import tensorflow as tf
import fishy_input
import numpy as np
import fishy_constants as const

FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def inputs(set_type):
  """
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  images, labels = fishy_input.inputs(data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size,
                                        set_type=set_type)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inference(images, reuse=None):
  """Build the model

  Args:
    images: Images returned from inputs()

  Returns:
    Logits.
  """

  # Convolution layer (64 outputs channels for every pixel)
  with tf.variable_scope('conv1', reuse=reuse) as scope:
    kernel = _variable_with_weight_decay('weights',
                                          shape=[5, 5, 3, 64],  # [filterHeight, filterWidth, in_channels, output_channels]
                                                                # => [filterHeight * filterWidth * in_channels, output_channels]
                                          stddev=5e-2,
                                          wd=0.0) 
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # norm
  norm = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm')

  # pool
  pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool')

  with tf.variable_scope('local1', reuse=reuse) as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # Softmax layer
  with tf.variable_scope('softmax_layer', reuse=reuse) as scope:
    weights = _variable_with_weight_decay('weights',
                                          shape=[384, const.NUM_CLASSES],
                                          stddev=5e-2,
                                          wd=0.0) 
    biases = _variable_on_cpu('biases', [const.NUM_CLASSES], tf.constant_initializer(0.0))

    logits = tf.add(tf.matmul(local1, weights), biases, name=scope.name)

    predictions = tf.nn.softmax(logits)

  return predictions, logits


def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  #tf.Graph.add_to_collections('losses', cross_entropy_mean)

  #return tf.Graph.add_n(tf.get_collection('losses'))
  return cross_entropy_mean


def train(total_loss):
  opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  grads = opt.compute_gradients(total_loss)

  apply_grad = opt.apply_gradients(grads)

  return apply_grad


def main(argv=None):
  print('Starting the session')

  with tf.Graph().as_default():

      xTrain, yTrain = inputs('Train')
      num_examples_per_epoch = fishy_input.get_input_length(FLAGS.data_dir, 'Train')

      predictions, logits = inference(xTrain)

      lss = loss(logits, yTrain)

      opt = train(lss)

      correct_prediction = tf.equal(tf.cast(yTrain, tf.int64), tf.argmax(predictions, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

      saver = tf.train.Saver()

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
      tf.summary.scalar('Training Cost', lss)
      tf.summary.scalar('Accuracy', accuracy)

      summary = tf.summary.merge_all()

      with tf.Session().as_default() as s:
        s.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=s, coord=coord)

        try:
            step = 0
            epoch = 0
            steps_per_epoch = num_examples_per_epoch / FLAGS.batch_size
            print('Epochs: '+ str(FLAGS.num_epochs) + '\tBatch_Size: ' + str(FLAGS.batch_size) + '\tSamples: ' + str(num_examples_per_epoch) + '\tSteps: ' + str(steps_per_epoch * FLAGS.num_epochs))

            while not coord.should_stop() and epoch < FLAGS.num_epochs:
                step += 1
                (summ, _) = s.run([summary, opt])
                (accur, cst) = s.run([accuracy, lss])

                writer.add_summary(summ, step * FLAGS.batch_size)
        
                #print(str(step))
                print(str(step) + '\tAccuracy = ' + str(round(accur, 3)) + '\tCost = ' +  str(cst))
                save_path = saver.save(s, '/tmp/fishy_model.ckpt')
                if step % steps_per_epoch == 0:
                    epoch += 1
                    print('\n-------------------------------------------------------------------------\n')
                    #print('\tCompleted epoch ' + str(epoch) + ' with a final cost of ' + str(evaluate()))
                    print('\tCompleted epoch ' + str(epoch)) 
                    print('\n-------------------------------------------------------------------------\n')



        except tf.errors.OutOfRangeError:
            print('out of range')
        finally:
            coord.request_stop()
            print('requesting stop')

if __name__ == '__main__':
  tf.app.run()

