import math
from data_handler import DataHandler
import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class Model:
    input_placeholder = None
    label_placeholder = None
    ns_placeholder = None
    ns_prob_placeholder = None
    embedding_variable = None
    loss = None

    optimizer = None

    session = None

    embedding_dimensions = 0
    vocabulary_size = 0
    negative_samples = 0
    learning_rate = 0

    def __init__(self, config, vocabulary_size):
        self.embedding_dimensions = config.embedding_dimensions
        self.negative_samples = config.negative_samples
        self.vocabulary_size = vocabulary_size
        self.learning_rate = config.learning_rate
        self.define_model()
        self.define_optimizer()
        self.start_session()

    def define_model(self):
        input_placeholder = tf.placeholder(tf.int32, shape=[1], name='input_token')  # (1)
        label_placeholder = tf.placeholder(tf.int32, shape=[1], name='label')  # (1)
        ns_placeholder = tf.placeholder(tf.int32,
                                        shape=[self.negative_samples], name='negative_samples')  # (neg_size)
        ns_prob_placeholder = tf.placeholder(tf.float64,
                                             shape=[self.negative_samples, 1], name='probabilities')  # (neg_size,1)

        with tf.variable_scope('word2vec'):
            embedding = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dimensions],
                                                      -1.0, 1.0, dtype=tf.float64))  # (v_size, e_size)
            weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_dimensions],
                                                      stddev=1.0/math.sqrt(self.embedding_dimensions), dtype=tf.float64))  # (v_size, e_size)

            input_embedding = tf.nn.embedding_lookup(embedding, input_placeholder)  # (e_size,)
            input_embedding = tf.reshape(input_embedding, [self.embedding_dimensions, 1])  # (e_size,1)

            label_tensor = tf.nn.embedding_lookup(weights, label_placeholder)  # (e_size,)
            label_tensor = tf.reshape(label_tensor, [self.embedding_dimensions, 1])  # (e_size, 1)

            neg_samples_tensor = tf.nn.embedding_lookup(weights, ns_placeholder)  # (neg_size, e_size)

            merge1 = tf.linalg.matmul(label_tensor, input_embedding, transpose_a=True)  # (1,1)
            merge2 = tf.linalg.matmul(neg_samples_tensor, input_embedding)  # (neg_size, 1)
            merge2 = tf.math.scalar_mul(-1.0, merge2)  # (neg_size, 1)

            label_sigmoid = tf.sigmoid(merge1)  # (1,1)
            label_log = tf.log(label_sigmoid)  # (1,1)

            neg_sample_sigmoid = tf.sigmoid(merge2)  # (neg_size, 1)
            neg_samples_log = tf.log(neg_sample_sigmoid)  # (neg_size, 1)

            expected_neg_sample = tf.linalg.matmul(neg_samples_log, ns_prob_placeholder, transpose_a=True)  # (1,1)

            result = tf.reduce_sum(label_log + expected_neg_sample)  # ()
            loss = tf.math.scalar_mul(-1, result)  # ()

            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
            normalized_embeddings = embedding / norm

            self.input_placeholder = input_placeholder
            self.label_placeholder = label_placeholder
            self.ns_placeholder = ns_placeholder
            self.ns_prob_placeholder = ns_prob_placeholder
            self.loss = loss
            self.embedding_variable = normalized_embeddings

    def define_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def start_session(self):
        init = tf.global_variables_initializer()
        self.session = tf.Session(config=tf_config)
        self.session.run(init)

    def end_session(self):
        self.session.close()

    def get_embeddings(self):
        return self.embedding_variable.eval(session=self.session)

    def run(self, data_handler: DataHandler, mode):
        total_loss_value = 0
        loss_value = 0
        total_samples = 0
        print_counter = 0

        while True:

            data = data_handler.get_next()
            if data is None:
                # data exhausted
                print("avg. " + mode + "-loss@step-" + str(total_samples) + " - " + str(loss_value / print_counter))
                return total_loss_value, total_samples

            for entry in data:
                feed_dict = {
                    self.input_placeholder: entry['word'],
                    self.label_placeholder: entry['label'],
                    self.ns_placeholder: entry['neg'],
                    self.ns_prob_placeholder: entry['prob']
                }
                if mode == "train":
                    _, loss = self.session.run([self.optimizer, self.loss], feed_dict)
                else:
                    [loss] = self.session.run([self.loss], feed_dict)

                loss_value += loss
                total_loss_value += loss

                total_samples += 1
                print_counter += 1

                if print_counter == 10000:
                    print("avg. " + mode + "-loss@step-" + str(total_samples) + " - " + str(loss_value / print_counter))
                    print_counter = 0
                    loss_value = 0
