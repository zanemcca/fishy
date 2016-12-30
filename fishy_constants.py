import os
import tensorflow as tf
import fishy_output as fo

CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
NUM_CLASSES = len(CLASSES) 

IMAGE_LIMIT = 2046 

#IMAGE_HEIGHT= 90
#IMAGE_WIDTH= 160
IMAGE_HEIGHT= 45
IMAGE_WIDTH= 80

NUM_EXAMPLES_PER_EPOCH = 32

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images per batch""")
tf.app.flags.DEFINE_integer('num_epochs', 1000, """Number of epochs to use for training""")
tf.app.flags.DEFINE_string('data_dir', os.getcwd(), """The data directory""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/fishy_log/' + str(fo.get_next_run_num('/tmp/fishy_log')), """The data directory""")
tf.app.flags.DEFINE_integer('use_fp16', False, """Train using floating point 16""")
tf.app.flags.DEFINE_float('learning_rate', 0.05, """Learning Rate""")
