from . import numpy as np
from . import tensorflow as tf
from . import numpy as np
from . import functools
from tensorflow.contrib.cluster_resolver.python.training.tpu_cluster_resolver import TPUClusterResolver

def do_jam_fw(tensor, name):
  filename = name + b'.bin'
  tensor.tofile(filename)
  return tensor

def do_jam_grad(op, grad):
  x = op.inputs[0]
  return x, None

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    #use default implementation of py_func if grad is None
    if grad == None:
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        output = tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    return output

def do_jam(tensor, name=""):
    print("tensor dumped as", name)
    out = py_func(do_jam_fw, [tensor, name], tensor.dtype, name=name, grad=do_jam_grad)
    out.set_shape(tensor.shape)
    return out

height = 1
width = 1
total_examples = 4

def predict_input_fn():
  def input_fn(params):
    batch_size = params['batch_size']

    images = tf.random_uniform(
               [total_examples, height, width, 4], minval=-1, maxval=1)
    labels = tf.random_uniform([total_examples], minval=-1, maxval=1)

    dataset0 = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset0 = dataset0.map(lambda images, lalels: ({'images': images}, {'labels': labels}))
    dataset0 = dataset0.batch(batch_size)

    """
    dataset1 = tf.data.Dataset.from_tensor_slices(labels)
    dataset1 = dataset1.map(lambda labels: {'labels': labels})
    dataset1 = dataset1.batch(batch_size)
    train_iterator1 = dataset1.make_initializable_iterator()
    labels = train_iterator1.get_next()
    """

    return dataset0


  return input_fn

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode, params):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float32)
  b = tf.get_variable("b", [1], dtype=tf.float32)

  conv1 = features['images']
#  conv1 = tf.cast(conv1, tf.float32)
  conv1 = tf.reshape(conv1, [-1,1,1,4])

  conv1 = tf.layers.conv2d(inputs=conv1, filters=1,
            kernel_size=[1, 1], padding="same", data_format='channels_first')
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=conv1)

  conv1 = do_jam(conv1, name="do_jam_conv")
  print(conv1.shape)

  conv1 = tf.reshape(conv1, [-1])
  print(conv1.shape)

  y = W*conv1 + b
  y = tf.reshape(y, [-1, 4])
  y = tf.Print(y, [tf.shape(y)], summarize=5, message="y shape:")
  print("y.shape:", y.shape, labels['labels'].shape)

  my_label = labels['labels']
  my_label = tf.Print(my_label, [tf.shape(my_label)], summarize=5, message="my_label shape:")

  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - my_label))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.estimator.EstimatorSpec(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

tpu_config = tf.contrib.tpu.RunConfig(master=TPUClusterResolver(tpu=[os.environ['TPU_NAME']]).get_master())
tpu_config.replace(model_dir='.')

#load customized model to estimator
estimator = tf.contrib.tpu.TPUEstimator(model_fn=model, config=tpu_config, use_tpu=True, predict_batch_size=4, train_batch_size=4)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

#input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=10, shuffle=False)
#eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1, shuffle=False)

# train
estimator.train(input_fn=predict_input_fn(), steps=10)
# Here we evaluate how well our model did.
#train_loss = estimator.evaluate(input_fn=predict_input_fn)
#eval_loss = estimator.evaluate(input_fn=eval_input_fn)
eval_pred = estimator.predict(input_fn=predict_input_fn())
#print("train loss: %r"% train_loss)
#print("eval loss: %r"% eval_loss)
print("print out eval prediction: ", eval_pred)
for i in eval_pred:
    print(i)
