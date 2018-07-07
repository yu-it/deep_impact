import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import common_api.common

is_local = True

def test_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant([[i for i in range(360)]],dtype=tf.float32)).map(lambda x:{"x":x})
    return dataset.make_one_shot_iterator()\
    .get_next()
    #return tf.data.TextLineDataset( r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\data\\training_input.csv").map(lambda x:{"x":tf.decode_csv(x,record_defaults=[tf.zeros([1],dtype=tf.float32)])}).make_one_shot_iterator().get_next()

def parse_csv(record):
    arr = tf.decode_csv(record,record_defaults=[tf.zeros([1],dtype=tf.float32),tf.zeros([1],dtype=tf.float32)])
    return {"x":arr[0]},arr[1]
def read_data_fn_old(input, label, repeatable=True):

    if repeatable:
        return tf.data.TextLineDataset("C:\\github\\deep_impact\\ml\\local\\experiment\\sample2_sin.csv")\
            .map(parse_csv)\
            .shuffle(5000)\
            .batch(100)\
            .repeat()\
            .make_one_shot_iterator()\
            .get_next()
    else:
        return tf.data.TextLineDataset("C:\\github\\deep_impact\\ml\\local\\experiment\\sample2_sin.csv")\
            .shuffle(5000)\
            .map(parse_csv)\
            .batch(100)\
            .make_one_shot_iterator()\
            .get_next()


def read_data_fn(input, label, repeatable=True):
    input_dataset = tf.data.TextLineDataset(input).map(lambda x:{"x":tf.decode_csv(x,record_defaults=[tf.zeros([1],dtype=tf.float32)])})
    label_dataset = tf.data.TextLineDataset(label).map(lambda x:tf.decode_csv(x,record_defaults=[tf.zeros([1],dtype=tf.float32)]))
    return_dataset = tf.data.Dataset.zip((input_dataset, label_dataset))
    if repeatable:
        return return_dataset.shuffle(5000)\
            .repeat()\
            .make_one_shot_iterator()\
            .get_next()
    else:
        return return_dataset.shuffle(5000)\
            .make_one_shot_iterator()\
            .get_next()

#
"""
DATA_INPUT_PATH = r"gs://deep_impact/ml-input/train_input.csv"
DATA_LABEL_PATH = r"gs://deep_impact/ml-input/train_label.csv"
EVAL_INPUT_PATH = r"gs://deep_impact/ml-input/eval_input.csv"
EVAL_LABEL_PATH = r"gs://deep_impact/ml-input/eval_label.csv"
MODEL_PATH = "gs://deep_impact/ml-input/sample2"
"""

DATA_INPUT_PATH = r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\data\\training_input.csv"
DATA_LABEL_PATH = r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\data\\training_label.csv"
EVAL_INPUT_PATH = r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\data\\eval_input.csv"
EVAL_LABEL_PATH = r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\data\\eval_label.csv"
MODEL_PATH = r"C:\\github\\deep_impact\\ml\\gcp\\sample2\\model"



input_fn = lambda:read_data_fn(DATA_INPUT_PATH, DATA_LABEL_PATH)
eval_fn = lambda:read_data_fn(EVAL_INPUT_PATH,EVAL_LABEL_PATH,False)

print(read_data_fn(DATA_INPUT_PATH,DATA_LABEL_PATH))
features = [tf.feature_column.numeric_column(key="x")]
#test_input(input_fn_lambda)


#8.79 model = tf.estimator.DNNRegressor(hidden_units=[10], feature_columns=feature_definition, model_dir=r"C:\\github\\deep_impact\\ml\\local\\experiment\\model")
#model = tf.estimator.DNNRegressor(activation_fn=tf.sigmoid,hidden_units=[20,20], optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001),feature_columns=feature_columns, model_dir=MODEL_PATH) #6.82
model = tf.estimator.DNNRegressor(activation_fn=tf.sigmoid,hidden_units=[20,20], optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001),feature_columns=features, model_dir=MODEL_PATH) #6.82
#model = tf.estimator.DNNRegressor(feature_columns=feature_definition, hidden_units=[1024, 512, 256],
#    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001))
#1,33
#model = tf.estimator.DNNRegressor(feature_columns=feature_definition, hidden_units=[256, 128, 64],
#    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001), model_dir=r"C:\\github\\deep_impact\\ml\\local\\experiment\\model")




model.train(input_fn=input_fn, steps=20000)

eval_result = model.evaluate(input_fn=eval_fn)
average_loss = eval_result["average_loss"]


print(eval_result )
predictions = model.predict(
    input_fn=test_fn)

y = []
for pred_dict in predictions:
    y.append(pred_dict['predictions'][0])

plt.plot(y)
plt.show()