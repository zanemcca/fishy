
import sys
import os
import math
import tensorflow as tf
import fishy
import numpy as np
import fishy_input
import fishy_constants as const

FLAGS = const.FLAGS

def evaluate(set_type='Test'):
  print('Starting the session')

  with tf.Graph().as_default():
    images, labels = fishy.inputs(set_type)
    num_examples_per_epoch = fishy_input.get_input_length(os.getcwd(),set_type)
    predictions,_ = fishy.inference(images)

    #lss = tf.contrib.losses.log_loss(predictions, tf.one_hot(labels, const.NUM_CLASSES), tf.truediv(1, FLAGS.batch_size), 1e-15) 
    lss = tf.contrib.losses.log_loss(predictions, tf.one_hot(labels, const.NUM_CLASSES), 1, 1e-15) 

    correct_prediction = tf.equal(tf.cast(labels, tf.int64), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
    tf.summary.scalar('Cost Evaluation', lss)
    tf.summary.scalar('Accuracy Evaluation', accuracy)

    summary = tf.summary.merge_all()

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
              (loss, acc) = s.run([lss, accuracy])

              summ = s.run(summary)

              writer.add_summary(summ, step * FLAGS.batch_size)

              """
              probs = predictions.eval()
              Y = labels.eval()

              cor = 0
              for i, y in enumerate(Y):
                top = np.argmax(probs[i])
                if(top == y):
                  cor += 1

                topChoices = [const.CLASSES[idx] for idx in np.argsort(probs[i])]
                topChoices.reverse()
                predic = ''
                for c in topChoices:
                  predic = predic + c + ','
                #print([round(p, 5) for p in probs[i]])
                prob = probs[i][y]
                cost = -math.log(max(min(prob, 1),1e-15))
                #print('(y, p): (' + const.CLASSES[y] + ', ' + predic + ')\tP(y): ' + str(round(prob, 2)) + '\tCost: ' + str(round(cost,2)))
                total = total + cost 

              print(str(step) + '\tAccuracy = ' + str(round(acc, 3)) + '\tPercentCorrect = ' + str(cor * 100 / FLAGS.batch_size) + '\tLoss = ' + str(loss) + '\tCost = ' +  str(total / (step * FLAGS.batch_size)))
              correct += cor
              """
              print(str(step) + '\tAccuracy = ' + str(round(acc, 3))+ '\tLoss = ' + str(loss))

              total += loss
              correct += acc

              if step % steps_per_epoch == 0:
                print('\nThe final Cost = ' + str(round(total / steps_per_epoch, 3)) + ' with a ' + str(round(correct * 100 / steps_per_epoch)) + '% of predictions being successful')
                return correct / steps_per_epoch 

      except tf.errors.OutOfRangeError:
          print('out of range evaluate')
      finally:
          coord.request_stop()
          print('requesting stop evaluate')

if __name__ == '__main__':
  evaluate('CV')
