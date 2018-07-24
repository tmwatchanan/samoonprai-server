# python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=C:\Users\Watchanan\PycharmProjects\samoonprai\herb_images\borapet\1.jpg

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import os

output_graph_file = os.path.join(os.getcwd(), "models", "image", "output_graph.pb")
output_labels_file = os.path.join(os.getcwd(), "models", "image", "output_labels.txt")

# Disable some logs
tf.logging.set_verbosity(tf.logging.WARN)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def classify_herb_image(file_name,
                        model_file=output_graph_file,
                        label_file=output_labels_file,
                        input_layer='Placeholder',
                        output_layer='final_result'):
    print("Classifying", file_name)
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255

    # if args.input_height:
    #     input_height = args.input_height
    # if args.input_width:
    #     input_width = args.input_width
    # if args.input_mean:
    #     input_mean = args.input_mean
    # if args.input_std:
    #     input_std = args.input_std
    # if args.input_layer:
    #     input_layer = args.input_layer
    # if args.output_layer:
    #     output_layer = args.output_layer

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    classification_output = []
    for i in top_k:
        out = {
            'label': labels[i],
            'score': results[i].item()
        }
        classification_output.append(out)
        # print(labels[i], results[i])
    print(classification_output)
    return (classification_output)
