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


"""
Returns:
  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
  labels: Labels. 1D tensor of [batch_size] size.

Raises:
  ValueError: If no data_dir
"""
def inputs(filename, limit=const.IMAGE_LIMIT):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  images, labels, names = fishy_input.inputs(filename=filename,
                                        batch_size=FLAGS.batch_size,
                                        limit=limit)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels, names



"""Build the model

Args:
  images: Images returned from inputs()

Returns:
  Logits.
"""
def inference(images, reuse=None, batch_size=FLAGS.batch_size):
  # Convolution layer (64 outputs channels for every pixel)
  with tf.variable_scope('conv1', reuse=reuse) as scope:
    kernel = _variable_with_weight_decay('weights',
                                          shape=[5, 5, const.IMAGE_CHANNELS, 64],  # [filterHeight, filterWidth, in_channels, output_channels]
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
    reshape = tf.reshape(pool, [batch_size, -1])
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

def get_global_step():
  with tf.device('/cpu:0'):
    try:
      with tf.variable_scope('global', reuse=False) as scope:
        _global_step = tf.get_variable('global_step', [], trainable=False, initializer=tf.constant_initializer(0.0))
    except ValueError:
      with tf.variable_scope('global', reuse=True) as scope:
        _global_step = tf.get_variable('global_step', [], trainable=False, initializer=tf.constant_initializer(0.0))

  return _global_step


def train(total_loss):
  global_step = get_global_step()

  opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  grads = opt.compute_gradients(total_loss)

  apply_grad = opt.apply_gradients(grads, global_step=global_step)

  return apply_grad

def main(argv=None):
  total_samples = 8000 
  min_epochs = 5

  with tf.Graph().as_default():
    num_examples = fishy_input.get_input_length('train.csv')
    cv_length = fishy_input.get_input_length('cv.csv')
    xCV, yCV, nameCV = inputs('cv.csv')

    X = tf.placeholder(tf.float32, name='X', shape=(FLAGS.batch_size, const.IMAGE_WIDTH, const.IMAGE_HEIGHT, const.IMAGE_CHANNELS))
    Y = tf.placeholder(tf.int32, name='Y', shape=(FLAGS.batch_size,))

    predictions, logits = inference(X)

    lss = loss(logits, Y)

    opt = train(lss)

    correct_prediction = tf.equal(tf.cast(Y, tf.int64), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    global_step = tf.cast(get_global_step(), tf.int64)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    name = FLAGS.name
    if name == str(const.LOG_NUMBER):
      name = str(const.EVAL_LOG_NUMBER)

    train_log = FLAGS.log_dir + '/train/' + name 
    cv_log = FLAGS.log_dir + '/cv/' + name

    cv_writer = tf.summary.FileWriter(cv_log, graph=tf.get_default_graph())
    train_writer = tf.summary.FileWriter(train_log, graph=tf.get_default_graph())

    Loss = tf.placeholder(tf.float32, name='Loss_Placeholder')
    Accuracy = tf.placeholder(tf.float32, name='Accuracy_Placeholder')
    tf.summary.scalar('Cost', Loss)
    tf.summary.scalar('Accuracy', Accuracy)

    summary = tf.summary.merge_all()

    for i in [10, 30, 100, 300, 500, 700]:
      xTrain, yTrain, nameTrain = inputs('train.csv', i)

      num_samples = i * const.NUM_CLASSES
      steps_per_epoch = math.ceil(float(num_samples) / FLAGS.batch_size)
      num_epochs = max(min_epochs, math.ceil(float(total_samples) / max(num_samples, FLAGS.batch_size)))

      with tf.Session().as_default() as s:
        s.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=s, coord=coord)

        try:
            epoch = 0

            # Training phase
            print('\nTraining...\n')
            print('Epochs: '+ str(num_epochs) + '\tBatch_Size: ' + str(FLAGS.batch_size) + '\tSamples: ' + str(num_samples) + '\tSteps: ' + str(steps_per_epoch * num_epochs) + '\n')
            step = 0
            training_cost = 0
            training_accuracy = 0
            last_error = 0
            while not coord.should_stop() and epoch < num_epochs:
              (x, y) = s.run([xTrain, yTrain])
              (step, accur, cst, _) = s.run([global_step, accuracy, lss, opt], feed_dict={ X: x, Y: y })

              save_path = saver.save(s, '/tmp/fishy_model.ckpt')

              if(epoch == num_epochs - 1):
                training_cost += cst
                training_accuracy += accur
      
              print('Step: ' + str(step) + ' \tAccuracy: ' + str(round(accur, 3)) + '\tCost: ' +  str(cst))
              if (step + 1) % steps_per_epoch == 0:
                  epoch += 1
                  print('') 

            training_cost /= steps_per_epoch
            training_accuracy /= steps_per_epoch
            summ = s.run(summary, feed_dict={ Loss: training_cost, Accuracy: training_accuracy })
            train_writer.add_summary(summ, num_samples)
            

            # Evaluation phase
            print('\nEvaluating...\n')
            steps_per_epoch = math.ceil(float(cv_length) / FLAGS.batch_size)
            eval_cost = 0
            eval_accuracy = 0
            stp = 0
            while not coord.should_stop():
              (x, y) = s.run([xCV, yCV])
              (accur, cst) = s.run([accuracy, lss], feed_dict={ X: x, Y: y })

              eval_cost += cst
              eval_accuracy += accur 
      
              stp += 1
              print('Evaluation - Accuracy: ' + str(round(accur, 3)) + '\tCost: ' +  str(cst))
              if stp == steps_per_epoch:
                eval_cost /= steps_per_epoch
                eval_accuracy /= steps_per_epoch
                summ = s.run(summary, feed_dict={ Loss: eval_cost, Accuracy: eval_accuracy })
                cv_writer.add_summary(summ, num_samples)
                print('\n-------------------------------------------------------------------------\n')
                print('\nEvaluation\tCost = ' + str(round(eval_cost, 3)) + ' with ' + str(round(eval_accuracy * 100, 1)) + '% accuracy')
                print('\nTraining\tCost = ' + str(round(training_cost, 3)) + ' with ' + str(round(training_accuracy * 100, 1)) + '% accuracy')
                print('\n-------------------------------------------------------------------------\n')
                break

                
        except tf.errors.OutOfRangeError:
            print('out of range')
        finally:
            coord.request_stop()
            print('requesting stop')


if __name__ == '__main__':
  tf.app.run()
