
import sys
import os
import math
import tensorflow as tf
import fishy
import numpy as np
import fishy_input
import fishy_constants as const
import fishy_output as out
from tensorflow.python.ops import math_ops

FLAGS = const.FLAGS

def evaluate(filename='submission_input.csv'):
  print('Starting the session')

  with tf.Graph().as_default():

    if not FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')

    num_examples_per_epoch = fishy_input.get_input_length(filename)
    batch_size = num_examples_per_epoch
    images, labels, names = fishy_input.inputs(filename=filename, batch_size=batch_size)

    predictions,_ = fishy.inference(images, batch_size=batch_size)
    
    lss = fishy.log_loss(predictions, labels)

    correct_prediction = tf.equal(tf.cast(labels, tf.int64), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    global_step = tf.cast(fishy.get_global_step(), tf.int64)

    saver = tf.train.Saver()

    with tf.Session() as s:
      saver.restore(s, "/tmp/fishy_model.ckpt")
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      try:
          step = 0
          steps_per_epoch = round(float(num_examples_per_epoch) / batch_size)
          print('Evaluating\tBatch_Size: ' + str(batch_size) + '\tSamples: ' + str(num_examples_per_epoch) + '\tSteps: ' + str(steps_per_epoch))

          total = 0
          correct = 0
          output = []
          completed_names = []
          while not coord.should_stop():
              step += 1
              (stp, loss, acc, files, probs) = s.run([global_step, lss, accuracy, names, predictions])
              
              for i in range(len(files)):
                name = os.path.basename(files[i])
                if(name not in completed_names):
                  completed_names.append(name)
                  data = {}
                  data['image'] = os.path.basename(files[i])
                  for j,v in enumerate(probs[i]):
                    data[const.CLASSES[j]] = v
                  output.append(data)
                else:
                  print(name)

              print(str(step) + '\tAccuracy = ' + str(round(acc, 3))+ '\tLoss = ' + str(loss))

              total += loss
              correct += acc

              if step == steps_per_epoch:
                print('\nThe final Cost = ' + str(round(total / steps_per_epoch, 3)) + ' with a ' + str(round(correct * 100 / steps_per_epoch)) + '% of predictions being successful')
                out.write_output(os.path.join('output', 'submission_' + FLAGS.name + '.csv'), output)
                return correct / steps_per_epoch 

      except tf.errors.OutOfRangeError:
          print('out of range evaluate')
      finally:
          coord.request_stop()
          print('requesting stop evaluate')

if __name__ == '__main__':
  evaluate('test.csv')
  #evaluate('submission_input.csv')
