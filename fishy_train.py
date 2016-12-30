
import tensorflow as tf
import fishy
import fishy_input
import fishy_constants as const

FLAGS = const.FLAGS

def train():
  print('Starting the session')

  with tf.Graph().as_default():

      xTrain, yTrain = fishy.inputs('Train')
      num_examples_per_epoch = fishy_input.get_input_length(FLAGS.data_dir, 'Train')

      predictions, logits = fishy.inference(xTrain)

      lss = fishy.loss(logits, yTrain)

      opt = fishy.train(lss)

      correct_prediction = tf.equal(tf.cast(yTrain, tf.int64), tf.argmax(predictions, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

      saver = tf.train.Saver()

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
      tf.summary.scalar('Cost Training', lss)
      tf.summary.scalar('Accuracy Training', accuracy)

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
  train()
  tf.app.run()

