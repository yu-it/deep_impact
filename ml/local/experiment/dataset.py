import tensorflow as tf
import numpy as np


def parse_csv(st):
    #ary = st.decode("utf-8").split(",")
    #st = tf.expand_dims(st, -1)
    ary = tf.decode_csv(st, record_defaults=[tf.zeros([1],dtype=tf.float32),tf.zeros([1],dtype=tf.float32)])
    return [{"x":ary[0]}, ary[1]]


dataset = tf.data.Dataset.from_tensor_slices(["sample2_sin.csv"])\
    .flat_map(lambda x : tf.data.TextLineDataset(x))\
    .map(parse_csv)
    #.map(lambda ary,ary2 : tf.cast(ary, dtype=tf.float32))
# .map(func_a) #\
print (dataset.output_types)
print (dataset.output_shapes)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.InteractiveSession()
for i in range(10):
    print(str(sess.run(next_element)))







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
    return dataset
    #return (dataset.shuffle(5000).batch(batch_size).repeat().make_one_shot_iterator().get_next())

#
"""
r = load_rawdata()
dataset2 = train_input_fn(r[0],r[1], 100)

next_element = dataset2.make_one_shot_iterator().get_next()
sess = tf.InteractiveSession()
for i in range(10):
    print(str(sess.run(next_element)))

"""



