import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route('/api/hello')
def hello():
    label_image()
    return render_template('hello.html')


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
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

def label_image():
    file_name = "data/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "data/retrained_graph.pb"
    label_file = "data/retrained_labels.txt"
    input_height = 128
    input_width = 128
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image", help="image to be processed")
    # parser.add_argument("--graph", help="graph/model to be executed")
    # parser.add_argument("--labels", help="name of file containing labels")
    # parser.add_argument("--input_height", type=int, help="input height")
    # parser.add_argument("--input_width", type=int, help="input width")
    # parser.add_argument("--input_mean", type=int, help="input mean")
    # parser.add_argument("--input_std", type=int, help="input std")
    # parser.add_argument("--input_layer", help="name of input layer")
    # parser.add_argument("--output_layer", help="name of output layer")
    # args = parser.parse_args()
    #
    # if args.graph:
    #     model_file = args.graph
    # if args.image:
    #     file_name = args.image
    # if args.labels:
    #     label_file = args.labels
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
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    for i in top_k:
        print(labels[i], results[i])