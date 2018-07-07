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


def get_statistics():
    return tf.get_default_graph().get_tensor_by_name(name="stdev_sepal_length:0"),
    tf.get_default_graph().get_tensor_by_name(name="stdev_sepal_width:0"),
    tf.get_default_graph().get_tensor_by_name(name="stdev_petal_length:0"),
    tf.get_default_graph().get_tensor_by_name(name="stdev_petal_width:0"),
    tf.get_default_graph().get_tensor_by_name(name="avg_sepal_length:0"),
    tf.get_default_graph().get_tensor_by_name(name="avg_sepal_width:0"),
    tf.get_default_graph().get_tensor_by_name(name="avg_petal_length:0"),
    tf.get_default_graph().get_tensor_by_name(name="avg_petal_width:0")




def parse_training_csv(record):
    iris_kind, \
    standard_sepal_length, \
    standard_sepal_width, \
    standard_petal_length, \
    standard_petal_width = tf.decode_csv(record,record_defaults=[ \
        tf.zeros([1], dtype=tf.int32), \
        tf.zeros([1],dtype=tf.float32),\
        tf.zeros([1],dtype=tf.float32),\
        tf.zeros([1],dtype=tf.float32),\
        tf.zeros([1],dtype=tf.float32),\
        ])
    return {"standard_sepal_length" : standard_sepal_length,
            "standard_sepal_width" : standard_sepal_width,
            "standard_petal_length" : standard_petal_length,
            "standard_petal_width" : standard_petal_width}\
           ,iris_kind


def parse_csv_without_label(record,stats,with_label = False):
    avg_sepal_length, \
    avg_sepal_width, \
    avg_petal_length, \
    avg_petal_width , \
    stdev_sepal_length, \
    stdev_sepal_width, \
    stdev_petal_length, \
    stdev_petal_width = [tf.constant(stats[0], name="avg_sepal_length", dtype = tf.float32),
    tf.constant(stats[1], name="avg_sepal_width", dtype = tf.float32),
    tf.constant(stats[2], name="avg_petal_length", dtype = tf.float32),
    tf.constant(stats[3], name="avg_petal_width", dtype = tf.float32),
    tf.constant(stats[4], name="stdev_sepal_length", dtype = tf.float32),
    tf.constant(stats[5], name="stdev_sepal_width", dtype = tf.float32),
    tf.constant(stats[6], name="stdev_petal_length", dtype = tf.float32),
    tf.constant(stats[7], name="stdev_petal_width", dtype = tf.float32)]

    if with_label:
        label,\
        raw_sepal_length, \
        raw_sepal_width, \
        raw_petal_length, \
        raw_petal_width,= tf.decode_csv(record, record_defaults=[ \
            tf.zeros([1], dtype=tf.int32), \
            tf.zeros([1], dtype=tf.float32), \
            tf.zeros([1], dtype=tf.float32), \
            tf.zeros([1], dtype=tf.float32), \
            tf.zeros([1], dtype=tf.float32), \

            ])

        return ({"standard_sepal_length": (raw_sepal_length - avg_sepal_length) / stdev_sepal_length,
                 "standard_sepal_width": (raw_sepal_width - avg_sepal_width) / stdev_sepal_width,
                 "standard_petal_length": (raw_petal_length - avg_petal_length) / stdev_petal_length,
                 "standard_petal_width": (raw_petal_width - avg_petal_width) / stdev_petal_width}, \
                label)

    else:
        raw_sepal_length, \
        raw_sepal_width, \
        raw_petal_length, \
        raw_petal_width = tf.decode_csv(record,record_defaults=[ \
            tf.zeros([1],dtype=tf.float32),\
            tf.zeros([1],dtype=tf.float32),\
            tf.zeros([1],dtype=tf.float32),\
            tf.zeros([1],dtype=tf.float32),\
            ])


        return {"standard_sepal_length" : (raw_sepal_length - avg_sepal_length) / stdev_sepal_length,
                "standard_sepal_width" : (raw_sepal_width - avg_sepal_width) / stdev_sepal_width,
                "standard_petal_length" : (raw_petal_length - avg_petal_length) / stdev_petal_length,
                "standard_petal_width" : (raw_petal_width - avg_petal_width) / stdev_petal_width}

def serving_fn(stats):
    """Build the serving inputs."""
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    features = parse_csv_without_label(csv_row,stats)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})

def training_data_reading_function(path):
    myset = tf.data.TextLineDataset(path)\
        .map(parse_training_csv)\
        .batch(10)
    return myset\
        .repeat() \
        .shuffle(200) \
        .make_one_shot_iterator()\
        .get_next()


def eval_data_and_statistics_reading_function(path,stats):
    myset = tf.data.TextLineDataset(path)\
        .map(lambda record:parse_csv_without_label(record,stats,True))\
        .batch(10)
    return myset\
        .make_one_shot_iterator()\
        .get_next()

def load_statistics(statistics_file):
    # Create a variable initialized to the value of a serialized numpy array
    f = StringIO(file_io.read_file_to_string(statistics_file))
    #my_variable = tf.Variable(initial_value=np.load(f), name='my_variable')
    return [float(x.replace('"',"")) for x in f.getvalue().split(",")]


#色々テストするためのフリースペース関数
def input_processor_test(args):
    sess = tf.InteractiveSession()
    data_reader = training_data_reading_function(args.train_files)
    print(data_reader)
    print(data_reader[0]["standard_petal_width"].eval())
    sess.close()


#色々テストするためのフリースペース関数
def sandbox():
    x = tf.constant([1, 4])
    print(x)
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
    tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        help='GCS or local paths,標準化された各種サイズとラベルをインプットにとる。',
        required=False,
        default=r"C:\\github\\deep_impact\\ml\\gcp\\sample3\\local_data\\training_data.csv"
    )
    parser.add_argument(
        '--eval-files',
        help='GCS or local paths,標準化されていない各種サイズをインプットにとる。',
        required=False,
        default=r"C:\\github\\deep_impact\\ml\\gcp\\sample3\\local_data\\eval_data.csv"
    )
    parser.add_argument(
        '--statistics-file',
        help='GCS or local paths,標準化に使用する統計値を格納したCSV',
        required=False,
        default=r"C:\\github\\deep_impact\\ml\\gcp\\sample3\\local_data\\statistics_data.csv"
    )
    parser.add_argument(
        '--output',
        help='GCS or local paths,学習したモデルの出力先',
        required=False,
        default=r"C:\\github\\deep_impact\\ml\\gcp\\sample3\\local_output"
    )
    parser.add_argument(
        '--job-dir',
        help='GCS or local paths,学習したモデルの出力先,outputとダブっているのでゆくゆく統合したいが、local trainモードとml engineモードでのパラメータの違いが・・・',
        required=False,
        default = r"C:\\github\\deep_impact\\ml\\gcp\\sample3\\local_output"
    )
    args = parser.parse_args()
    #input_processor_test(args)
    input_fn = lambda:training_data_reading_function(args.train_files)
    stats = load_statistics(args.statistics_file)
    eval_fn = lambda:eval_data_and_statistics_reading_function(args.eval_files,stats)
    serving_fn_lambda = lambda:serving_fn(stats)



    features = [tf.feature_column.numeric_column(key="standard_sepal_length"),
            tf.feature_column.numeric_column(key="standard_sepal_width"),
            tf.feature_column.numeric_column(key="standard_petal_length"),
            tf.feature_column.numeric_column(key="standard_petal_width")
            ]
    model = tf.estimator.DNNClassifier( feature_columns=features,
                                         hidden_units=[256, 32],
                                         optimizer=tf.train.AdamOptimizer(1e-4),
                                         n_classes=3,
                                         dropout=0.1,
                                         model_dir=args.output) #6.82



    train_spec = tf.estimator.TrainSpec(input_fn,
                                        max_steps=20000

                                        )
    exporter = tf.estimator.FinalExporter('iris',
                                          serving_fn_lambda)
    eval_spec = tf.estimator.EvalSpec(eval_fn,
                                      steps=10,
                                      exporters=[exporter],
                                      name='iris-eval',

                        )
    #eval_spec = tf.estimator.EvalSpec(eval_fn,
    #                                  steps=10,
    #                                  name='iris-eval',
    #                                   hooks=[MyHook(args.statistics_file)]
    #                                  )

    tf.estimator.train_and_evaluate(model,
                                    train_spec,
                                    eval_spec)

    #test
    predictions = model.predict(
        input_fn=eval_fn)

    y = []
    for pred_dict in predictions:
        print(pred_dict['class_ids'])

    #plt.plot(y)
    #plt.show()
