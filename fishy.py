import os
import tensorflow as tf
import fishy_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images per batch""")
tf.app.flags.DEFINE_integer('num_epochs', 5, """Number of epochs to use for training""")
tf.app.flags.DEFINE_string('data_dir', os.getcwd(), """The data directory""")
tf.app.flags.DEFINE_integer('use_fp16', False, """Train using floating point 16""")
tf.app.flags.DEFINE_float('learning_rate', 0.1, """Learning Rate""")

NUM_CLASSES = 8

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
  tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


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



def inputs(eval_data):
  """
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  images, labels, imgCnt = fishy_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels, imgCnt

def inference(images):
  """Build the model

  Args:
    images: Images returned from inputs()

  Returns:
    Logits.
  """

  # Convolution layer
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                          shape=[5, 5, 3, 64],
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

  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # Softmax layer
  with tf.variable_scope('softmax_layer') as scope:
    weights = _variable_with_weight_decay('weights',
                                          shape=[384, NUM_CLASSES],
                                          stddev=5e-2,
                                          wd=0.0) 
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)

  return softmax_linear


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

      xTrain, yTrain, num_examples_per_epoch = inputs(False)

      logits = inference(xTrain)

      lss = loss(logits, yTrain)

      opt = train(lss)

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      s = tf.Session()

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
              (y, cst, _) = s.run([logits, lss, opt])

              print(str(step) + '\tCost = ' +  str(cst))
              #X, Y = s.run([xTrain, yTrain])
              #print(str(step), 'X=', str(X), 'Y=', str(Y))
              #print(str(step), 'Y=', str(Y))
              if step % steps_per_epoch == 0:
                  epoch += 1
                  print('\n--------------------------------------------------\n')
                  print(' Completed the ' + str(epoch) + ' epoch')
                  print('\n--------------------------------------------------\n')

      except tf.errors.OutOfRangeError:
          print('out of range')
      finally:
          coord.request_stop()
          print('requesting stop')

if __name__ == '__main__':
  tf.app.run()

