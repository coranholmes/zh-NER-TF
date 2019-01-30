import tensorflow as tf
from data import *


def main():
    batch_size = 4
    filename_queue = tf.train.string_input_producer(["data_path/test_data"],  shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, num_records=batch_size)
    batch_values = tf.train.batch([value], batch_size=batch_size, capacity=64000, enqueue_many=True)

    with tf.Session() as sess:
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(4):
            num_examples = 0
            try:
                for i in range(4):
                    data = sess.run([batch_values])
                    data = read_corpus(data)
                    for (seqs, labels) in data:
                        print(seqs)
                        print(labels)
                    num_examples += 1
            except tf.errors.OutOfRangeError:
                print ("There are", num_examples, "samples")

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
