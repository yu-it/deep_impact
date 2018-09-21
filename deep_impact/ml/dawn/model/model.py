# -*- coding: utf-8 -*-
import sys
import deep_impact.ml.vector_parser as vector_parser
import deep_impact.ml.dawn.model.vector_definition as vector_definition

if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

import tensorflow as tf


def input_processor_test_new_func(args):
    sess = tf.InteractiveSession()
    filter_func = lambda line: tf.less_equal(tf.string_to_number(tf.substr(line, 0, 1), tf.int32), 6)

    vector_def = vector_parser.vector_parser(vector_definition.definition_for_test, True)
    #data_reader = training_data_reading_function(args.train_files)
    myset = tf.data.TextLineDataset(args.train_files)\
        .skip(1) \
        .filter(lambda line: tf.less_equal(tf.string_to_number(tf.substr(line, 0, 1), tf.int32), 6)) \
        .map(vector_def)\
        .batch(100000)
    data_reader = myset\
        .make_one_shot_iterator()\
        .get_next()
    #data_reader = training_data_reading_function(args.train_files)
    print(data_reader)
    print(len(data_reader[0]["v_24"].eval()))  # print(data_reader[0]["v_0"].eval())
    sess.close()

training_csv_parser = vector_parser.vector_parser(vector_definition.definition, True)
predict_service_csv_parser = vector_parser.vector_parser(vector_definition.definition, False)

def serving_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    features = predict_service_csv_parser(csv_row)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})

def training_data_reading_function(path):
    #
    myset = tf.data.TextLineDataset(path)\
        .skip(1) \
        .filter(lambda line: tf.less_equal(tf.string_to_number(tf.substr(line, 0, 1), tf.int32), 6)) \
        .map(training_csv_parser)\
        .batch(10)
    return myset\
        .repeat() \
        .shuffle(20) \
        .make_one_shot_iterator()\
        .get_next()
def eval_data_reading_function(path):
    myset = tf.data.TextLineDataset(path)\
        .skip(1) \
        .filter(lambda line: tf.greater(tf.string_to_number(tf.substr(line, 0, 1), tf.int32), 6)) \
        .map(training_csv_parser)\
        .batch(10)
    return myset\
        .make_one_shot_iterator()\
        .get_next()



def train(args):
    input_fn = lambda:training_data_reading_function(args.train_files)
    eval_fn = lambda:eval_data_reading_function(args.eval_files)
    serving_fn_lambda = lambda:serving_fn()

    features = vector_parser.extract_features(vector_definition.definition)

    model = tf.estimator.DNNClassifier(feature_columns=features, hidden_units=[2048, 512],
                                       optimizer=tf.train.AdamOptimizer(1e-4), n_classes=19, activation_fn=tf.sigmoid,
                                       dropout=0.3, model_dir=args.output)  # 6.82

    train_spec = tf.estimator.TrainSpec(input_fn,
                                        max_steps=50000
                                        )
    exporter = tf.estimator.FinalExporter('dawn',
                                          serving_fn_lambda)
    eval_spec = tf.estimator.EvalSpec(eval_fn,
                                      steps=10,
                                      #exporters=[exporter],
                                      name='dawn-eval',

                        )
    #eval_spec = tf.estimator.EvalSpec(eval_fn,
    #                                  steps=10,
    #                                  name='iris-eval',
    #                                   hooks=[MyHook(args.statistics_file)]
    #                                  )

    eval = tf.estimator.train_and_evaluate(model,
                                    train_spec,
                                    eval_spec)

    import datetime
    print(datetime.datetime.now())
    #test
    predictions = model.predict(
        input_fn=eval_fn)

    y = []
    for pred_dict in predictions:
        try:
            print(pred_dict['class_ids'])
        except:
            print("-")

    #eval_result = model.evaluate(
    #    input_fn=eval_fn)

    #print (eval_result)
    #plt.plot(y)
    #plt.show()
