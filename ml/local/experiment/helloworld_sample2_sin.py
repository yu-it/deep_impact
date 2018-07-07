import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

test_predict_x = {
    'x': [i for i in range(360)]
}


def load_rawdata():
    source = []
    with open("sample2_sin.csv", "r") as f:
        for line in f:
            source.append(line.split(","))

    data = np.array(source, dtype="float").transpose()
    feature = data[0]
    label = [x for x in data[1]]
    return [{"x":feature},label]

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    print(dataset)

    return (dataset.shuffle(5000).batch(batch_size).repeat().make_one_shot_iterator().get_next())
    # Shuffle, repeat, and batch the examples.
    #return dataset.shuffle(1000).repeat().batch(batch_size)


def define_feature_columns(dataset):
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key="x"))
    return my_feature_columns

def eval_input_fn(features, labels, batch_size):
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


rawdata = load_rawdata()

input_fn = lambda:train_input_fn(rawdata[0],rawdata[1], 100)
eval_fn = lambda:eval_input_fn(rawdata[0],rawdata[1], 30)
test_fn = lambda:eval_input_fn(test_predict_x,None, batch_size=30)

feature_definition = define_feature_columns(input_fn)

#8.79 model = tf.estimator.DNNRegressor(hidden_units=[10], feature_columns=feature_definition, model_dir=r"C:\\github\\deep_impact\\ml\\local\\experiment\\model")
model = tf.estimator.DNNRegressor(activation_fn=tf.sigmoid,hidden_units=[20,20], optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001),feature_columns=feature_definition, model_dir=r"C:\\github\\deep_impact\\ml\\local\\experiment\\model") #6.82
#model = tf.estimator.DNNRegressor(feature_columns=feature_definition, hidden_units=[1024, 512, 256],
#    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001))
#1,33
#model = tf.estimator.DNNRegressor(feature_columns=feature_definition, hidden_units=[256, 128, 64],
#    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001), model_dir=r"C:\\github\\deep_impact\\ml\\local\\experiment\\model")



print("before train")

model.train(input_fn=input_fn, steps=20000)

print("before eval")
#eval_result = model.evaluate(input_fn=eval_fn)
#average_loss = eval_result["average_loss"]
#print("before test")

#print(eval_result)

predictions = model.predict(
    input_fn=test_fn)

print("a")


y = []
for pred_dict in predictions:
    y.append(pred_dict['predictions'][0])

plt.plot(y)
plt.show()