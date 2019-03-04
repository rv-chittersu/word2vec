import math
from input_handler import DataHandler
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_model(neg_samples, embedding_dim, vocab_size):
    input_token = tf.placeholder(tf.int32, shape=[1], name='input_token')  # (1)
    label = tf.placeholder(tf.int32, shape=[1], name='label')  # (1)
    neg_sample_enc = tf.placeholder(tf.int32, shape=[neg_samples], name='negative_samples')  # (neg_size)
    probabilities = tf.placeholder(tf.float32, shape=[neg_samples, 1], name='probabilities')  # (neg_size,1)

    with tf.variable_scope('word2vec'):
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))  # (v_size, e_size)
        weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim],
                                                  stddev=1.0/math.sqrt(embedding_dim)))  # (v_size, e_size)

        input_embedding = tf.nn.embedding_lookup(embedding, input_token)  # (e_size,)
        input_embedding = tf.reshape(input_embedding, [embedding_dim, 1])  # (e_size,1)

        label_tensor = tf.nn.embedding_lookup(weights, label)  # (e_size,)
        label_tensor = tf.reshape(label_tensor, [embedding_dim, 1])  # (e_size, 1)

        neg_samples_tensor = tf.nn.embedding_lookup(weights, neg_sample_enc)  # (neg_size, e_size)

        merge1 = tf.linalg.matmul(label_tensor, input_embedding, transpose_a=True)  # (1,1)
        merge2 = tf.linalg.matmul(neg_samples_tensor, input_embedding)  # (neg_size, 1)
        merge2 = tf.math.scalar_mul(-1.0, merge2)  # (neg_size, 1)

        label_sigmoid = tf.sigmoid(merge1)  # (1,1)
        label_log = tf.log(label_sigmoid)  # (1,1)

        neg_sample_sigmoid = tf.sigmoid(merge2)  # (neg_size, 1)
        neg_samples_log = tf.log(neg_sample_sigmoid)  # (neg_size, 1)

        expected_neg_sample = tf.linalg.matmul(neg_samples_log, probabilities, transpose_a=True)  # (1,1)

        result = label_log + expected_neg_sample  # (1,1)
        loss = tf.math.scalar_mul(-1, result)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embeddings = embedding / norm

        return input_token, label, neg_sample_enc, probabilities, normalized_embeddings, loss


def run(input_enc, label, neg_samples, prob, embeddings, loss, data_handler: DataHandler):
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    average_loss = 0
    training_step = 0
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as session:
        session.run(init)

        for i in range(0, 3):
            print("==========================")
            print("Starting epoch:" + str(i))
            print("==========================")
            while True:
                data = data_handler.get_next()
                if data is None:
                    # report final results
                    data_handler.reset()
                    break
                for entry in data:
                    feed_dict = {
                        input_enc: entry['word'],
                        label: entry['label'],
                        neg_samples: entry['neg'],
                        prob: entry['prob']
                    }

                    training_step += 1

                    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val

                    if training_step % 10000 == 0:
                        print("Avg.loss@" + str(training_step) + " - " + str(average_loss))
                        average_loss = 0

        final_embeddings = embeddings.eval()
        return final_embeddings
