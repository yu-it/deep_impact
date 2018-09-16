# -*- coding: utf-8 -*-
# C:\github\deep_impact\ml\gcp\sample3に配置して実行する。
"""
training-input   標準化された各種サイズのCSV,レイアウト:ラベル,sepal_length,sepal_width,,petal_length,petal_width
prediction-input 標準化されていない各種サイズのCSV,,レイアウト:sepal_length,sepal_width,,petal_length,petal_width
                 →統計値をテンソルとしてsaved_model内に保存できるかを試す、および結果が適切であることを検証しやすいように
                 　あえてtrainingとpredictで違うタイプの入力を与える形で実装
output           アヤメ種別
"""
import sys

if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io

import argparse
#import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#import common_api.common
def parse_csv_to_vector(record,with_label = True):
    vector_definition = []
    vector_definition.append(tf.zeros([1],dtype=tf.int32))
    vector_definition.append(tf.zeros([1],dtype=tf.string))
    vector_definition.extend([tf.zeros([1],dtype=tf.float32) for x in range(545)])
    vector_definition.append(tf.zeros([1],dtype=tf.int32))
    csv_vectors = tf.decode_csv(record,record_defaults=vector_definition)
    input_dict = {}
    [input_dict.update({"v_{num}".format(num=i) : v}) for (i,v) in enumerate(csv_vectors[2 : 2 + 545])]
    if with_label:
        return input_dict,csv_vectors[-1]
    else:
        return input_dict

def serving_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    features = parse_csv_to_vector(csv_row,False)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})

def training_data_reading_function(path):
    #
    myset = tf.data.TextLineDataset(path)\
        .skip(1) \
        .filter(lambda line: tf.less_equal(tf.string_to_number(tf.substr(line, 0, 1), tf.int32), 6)) \
        .map(parse_csv_to_vector)\
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
        .map(lambda record:parse_csv_to_vector(record,True))\
        .batch(10)
    return myset\
        .make_one_shot_iterator()\
        .get_next()


def input_processor_test(args):
    sess = tf.InteractiveSession()
    data_reader = eval_data_reading_function(args.train_files)
    print(data_reader)
    for idx in range(10000):
        print("seq {idx} size{size})".format(idx = idx, size = len(data_reader[0]["v_24"].eval())))
    print (data_reader[0]["v_24"].eval() )  #print(data_reader[0]["v_0"].eval())
    sess.close()

def train(args):
    input_fn = lambda:training_data_reading_function(args.train_files)
    eval_fn = lambda:eval_data_reading_function(args.eval_files)
    serving_fn_lambda = lambda:serving_fn()

    features = [tf.feature_column.numeric_column(key="v_{num}".format(num = i)) for i in range(545)]

    model = tf.estimator.DNNClassifier( feature_columns=features,
                                         hidden_units=[2048, 512],
                                         optimizer=tf.train.AdagradOptimizer(0.1),
                                         n_classes=19,
                                         dropout=0.3,
                                        activation_fn=tf.sigmoid,
                                         model_dir=args.output) #6.82

    train_spec = tf.estimator.TrainSpec(input_fn,
                                        max_steps=10000
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
