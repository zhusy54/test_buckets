import numpy as np
import tensorflow as tf

import numpy as np

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
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def do_jam(tensor, name=""):
    print("tensor dumped as", name)
    out = py_func(do_jam_fw, [tensor, name], [tensor.dtype], name=name, grad=do_jam_grad)
    return out
