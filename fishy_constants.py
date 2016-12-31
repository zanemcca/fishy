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
IMAGE_CHANNELS = 1 

NUM_EXAMPLES_PER_EPOCH = 32

LOG_NUMBER = fo.get_next_run_num('/tmp/fishy_log')
EVAL_LOG_NUMBER = max(fo.get_next_run_num('/tmp/fishy_log/train'), fo.get_next_run_num('/tmp/fishy_log/cv'))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images per batch""")
tf.app.flags.DEFINE_integer('num_epochs', 10, """Number of epochs to use for training""")
tf.app.flags.DEFINE_string('data_dir', os.getcwd(), """The data directory""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/fishy_log', """The log directory""")
tf.app.flags.DEFINE_string('name', str(LOG_NUMBER), """The name of the run""")
tf.app.flags.DEFINE_integer('use_fp16', False, """Train using floating point 16""")
tf.app.flags.DEFINE_float('learning_rate', 0.05, """Learning Rate""")
