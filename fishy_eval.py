
import sys
import os
import math
import tensorflow as tf
import fishy
import numpy as np
import fishy_input


FLAGS = tf.app.flags.FLAGS
CATEGORIES = fishy_input.CATEGORIES

def evaluate(set_type='Test'):
  print('Starting the session')

  with tf.Graph().as_default():
    images, labels = fishy.inputs(set_type)
    num_examples_per_epoch = fishy_input.get_input_length(os.getcwd(),set_type)
    logits = fishy.inference(images)

    predictions = tf.nn.softmax(logits)

    saver = tf.train.Saver()

    with tf.Session() as s:
      saver.restore(s, "/tmp/fishy_model.ckpt")
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      try:
          step = 0
          steps_per_epoch = round(float(num_examples_per_epoch) / FLAGS.batch_size)
          print('Evaluating\tBatch_Size: ' + str(FLAGS.batch_size) + '\tSamples: ' + str(num_examples_per_epoch) + '\tSteps: ' + str(steps_per_epoch))

          total = 0
          correct = 0
          while not coord.should_stop():
              step += 1
              probs= predictions.eval()
              Y = labels.eval()

              cor = 0
              for i, y in enumerate(Y):
                top = np.argmax(probs[i])
                if(top == y):
                  cor += 1

                topChoices = [CATEGORIES[idx] for idx in np.argsort(probs[i])]
                topChoices.reverse()
                predic = ''
                for c in topChoices:
                  predic = predic + c + ','
                #print([round(p, 5) for p in probs[i]])
                prob = probs[i][y]
                cost = -math.log(max(min(prob, 1),1e-15))
                #print('(y, p): (' + CATEGORIES[y] + ', ' + predic + ')\tP(y): ' + str(round(prob, 2)) + '\tCost: ' + str(round(cost,2)))
                total = total + cost 

              print(str(step) + '\tPercentCorrect = ' + str(cor * 100 / FLAGS.batch_size) + '\tCost = ' +  str(total / (step * FLAGS.batch_size)))
              correct += cor
              if step % steps_per_epoch == 0:
                print('\nThe final Cost = ' + str(round(total / num_examples_per_epoch, 2)) + ' with a ' + str(correct * 100 / num_examples_per_epoch) + '% of predictions being successful')
                return total / num_examples_per_epoch

      except tf.errors.OutOfRangeError:
          print('out of range evaluate')
      finally:
          coord.request_stop()
          print('requesting stop evaluate')

if __name__ == '__main__':
  evaluate('Train')
