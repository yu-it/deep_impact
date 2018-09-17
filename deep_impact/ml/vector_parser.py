import tensorflow as tf
class vector_entry:
    class usage:
        ForInput = "Input"
        ForLabel = "Label"
        Discard = "Discard"
    def __init__(self, name, tftype, usage):
        self.name = name
        self.tftype = tftype
        self.usage = usage


class vector_parser:

    def __init__(self,vector_list,with_label):
        self.vector_list = vector_list
        self.with_label = with_label

    def __call__(self, record):
        csv_definition = [tf.zeros([1], dtype=vec_list.tftype) for vec_list in self.vector_list]

        name_index = {}
        [name_index.update({vec_entry.name: idx}) for idx,vec_entry in enumerate(self.vector_list)]

        csv_vectors = tf.decode_csv(record, record_defaults=csv_definition)
        input_dict = {}
        [input_dict.update({vec_entry.name: csv_vectors[name_index[vec_entry.name]]}) for vec_entry in filter(lambda x: x.usage == vector_entry.usage.ForInput, self.vector_list)]
        if self.with_label:

            return input_dict, [csv_vectors[name_index[vec_entry.name]] for vec_entry in filter(lambda x: x.usage == vector_entry.usage.ForLabel, self.vector_list)][0]
        else:
            return input_dict


def extract_features(vector_list):
    return [tf.feature_column.numeric_column(key=vec_entry.name) for vec_entry in
     filter(lambda x: x.usage == vector_entry.usage.ForInput, vector_list)]
