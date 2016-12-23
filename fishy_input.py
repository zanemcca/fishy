
import os
import random
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from PIL import Image

FLAGS = tf.app.flags.FLAGS

IMAGE_LIMIT = -1 

IMAGE_HEIGHT= 90
IMAGE_WIDTH= 160

CATEGORIES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

NUM_CLASSES = len(CATEGORIES)
NUM_EXAMPLES_PER_EPOCH = 32

NUM_EPOCHS = 3 
BATCH_SIZE = 32

def read_image(filename_queue):
  label = filename_queue[1] 

  image = tf.read_file(filename_queue[0])
  img = tf.image.decode_jpeg(image, 3)

  return img, label 

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def tfPrint(tensor):
  with tf.Session() as sess:
    #t = tf.constant(tensor)
    print(sess.run(tensor))
    sess.close()

def inputs(eval_data, data_dir, batch_size):
  """
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  if not eval_data:
    data_dir = os.path.join(data_dir, 'train')
  else:
    data_dir = os.path.join(data_dir, 'test_stg1')

  print('Fetching file names...')

  filenames = []
  for (dirpath, dirnames, files) in os.walk(data_dir):
    filenames.extend([os.path.join(dirpath, f) for f in files if ('.jpg' in f or '.jpeg' in f)])

  print('Found ' + str(len(filenames)) + ' filenames')
  if(len(filenames) > IMAGE_LIMIT and IMAGE_LIMIT > 0):
    print('Taking a random sampling of ' + str(IMAGE_LIMIT) + ' images')
    filenames = random.sample(filenames, IMAGE_LIMIT)

  labels = []
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
    else:
      label =-1 
      for i, c in enumerate(CATEGORIES):
        if c in f:
          label = i
          break

      if(label >= 0):
        labels.append(label)
      else:
        raise ValueError('No label found for ' + f)

  global NUM_EXAMPLES_PER_EPOCH 
  NUM_EXAMPLES_PER_EPOCH = len(labels) 

  # Create a queue that produces the filenames to read.
  filenames = tf.convert_to_tensor(filenames, dtype=dtypes.string)
  labels = tf.convert_to_tensor(labels, dtype=dtypes.int32)

  filename_queue = tf.train.slice_input_producer([filenames, labels])

  read_input, label = read_image(filename_queue)

  # Read examples from files in the filename queue.
  #reshaped_image = tf.image.convert_image_dtype(read_input, dtype=tf.float32)
  reshaped_image = tf.cast(read_input, tf.float32)

  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  #resized_image = rp.resize_image_with_crop_or_pad(reshaped_image,width, height)
  resized_image = tf.image.resize_images(reshaped_image,[width, height])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  #min_queue_examples = int(num_examples_per_epoch *
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  (imgs, lbls) =  _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         #shuffle=True), NUM_EXAMPLES_PER_EPOCH
                                         shuffle=True)
  return imgs, lbls, NUM_EXAMPLES_PER_EPOCH


def test_input():
  print('Starting the session')

  with tf.Graph().as_default():
      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      xTrain, yTrain, _ = inputs(False, os.getcwd(), BATCH_SIZE)

      s = tf.Session()

      s.run(init_op)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=s, coord=coord)

      try:
          step = 0
          epoch = 0
          steps_per_epoch = NUM_EXAMPLES_PER_EPOCH / BATCH_SIZE
          print('Epochs: '+ str(NUM_EPOCHS) + '\tBatch_Size: ' + str(BATCH_SIZE) + '\tSamples: ' + str(NUM_EXAMPLES_PER_EPOCH) + '\tSteps: ' + str(steps_per_epoch * NUM_EPOCHS))

          while not coord.should_stop() and epoch < NUM_EPOCHS:
              X, Y = s.run([xTrain, yTrain])
              step += 1
              #print(str(step), 'X=', str(X), 'Y=', str(Y))
              print(str(step), 'Y=', str(Y))
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
  test_input()

