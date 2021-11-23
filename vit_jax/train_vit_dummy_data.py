# On Cloud VM, run
"""
pip install --upgrade pip
export PATH=/home/yfeng_us/.local/bin:${PATH}

pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip uninstall -y six typing-extensions tf-nightly
pip install tensorflow==2.7.0 flax einops tensorflow_datasets

# Clone repository and pull latest changes.
rm -rf vision_transformer || true
git clone --depth=1 https://github.com/yf225/vision_transformer -b vit_dummy_data
export PYTHONPATH=/home/yfeng_us/vision_transformer:${PYTHONPATH}
cd vision_transformer/

python3 vit_jax/train_vit_dummy_data.py
"""

# References:
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax.ipynb
# - https://github.com/google/flax/blob/main/examples/imagenet/train.py

# Google Colab "TPU" runtimes are configured in "2VM mode", meaning that JAX
# cannot see the TPUs because they're not directly attached. Instead we need to
# setup JAX to communicate with a second machine that has the TPUs attached.
import jax
import os
if 'COLAB_TPU_ADDR' in os.environ:
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
assert "tpu" in str(jax.local_devices()[0]).lower()
assert jax.local_device_count() == 8
print('Connected to TPU.')

import functools
import os
import time

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

DEBUG = False

# Hyperparams
num_attention_heads = 16
hidden_size = 1280
num_layers = 32

micro_batch_size = 8  # batch size per TPU core
print("micro_batch_size: ", micro_batch_size)

model_dtype = jnp.bfloat16 # jnp.float32
input_dtype = tf.bfloat16 # tf.float32
if model_dtype == jnp.float32:
  opt_dtype = 'float32'
elif model_dtype == jnp.bfloat16:
  opt_dtype = 'bfloat16'
else:
  raise Exception("Unknown model_dtype: {}".format(model_dtype))

if DEBUG:
  print("Overwriting hyperparams since we are in DEBUG mode...")
  num_attention_heads = 1
  hidden_size = 128
  num_layers = 1
  micro_batch_size = 1  # batch size per TPU core

num_steps = 50
accum_steps = 1  # How many steps to accumulate gradients for, before the gradient update
learning_rate = 0.001

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
num_classes = 1000
dropout_rate = 0.

import sys
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')

# Clone repository and pull latest changes.
#!rm -rf vision_transformer || true
#!git clone --depth=1 https://github.com/yf225/vision_transformer -b vit_dummy_data
#!cd vision_transformer && git pull
# %load_ext autoreload
# %autoreload 2

from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip
from vit_jax import utils


def make_update_fn(*, apply_fn, accum_steps, lr_fn):
  """Returns update step for data parallel training."""

  def update_fn(opt, step, batch, rng):

    _, new_rng = jax.random.split(rng)
    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=dropout_rng),
          inputs=images,
          train=True)
      return cross_entropy_loss(logits=logits, labels=labels)

    l, g = utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), opt.target, batch[0], batch[1],
        accum_steps)
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    l = jax.lax.pmean(l, axis_name='batch')

    opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
    return opt, l, new_rng

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def get_random_data(*, num_classes,
             image_size, global_batch_size, num_steps):
  num_devices = jax.local_device_count()

  data = tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(np.random.randn(1, global_batch_size, image_size, image_size, 3) , dtype=input_dtype),
    # tf.one_hot(np.random.randint(0, num_classes, size=(1, global_batch_size, 1)), num_classes),
    tf.one_hot(np.zeros((1, global_batch_size, 1)), num_classes),
  ))

  # Shard data such that it can be distributed accross devices
  def _shard(data_image, data_label):
    data_image = tf.reshape(data_image,
                               [num_devices, -1, image_size, image_size, 3])
    data_label = tf.reshape(data_label,
                               [num_devices, -1, num_classes])
    return data_image, data_label

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.repeat(num_steps + 1).prefetch(2)


def train():
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  global_batch_size = micro_batch_size * jax.local_device_count()
  ds_train = get_random_data(num_classes=num_classes, image_size=image_size, global_batch_size=global_batch_size, num_steps=num_steps)
  batch = next(iter(ds_train))
  print(batch[0].shape, batch[1].shape)

  # Build VisionTransformer architecture
  model = models.VisionTransformer(
    num_heads=num_attention_heads,
    hidden_size=hidden_size,
    num_layers=num_layers,
    patch_size=patch_size,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    dtype=model_dtype,
  )

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch[0].shape[1:], model.dtype),
        train=False)

  # This compiles the model to XLA (takes some minutes the first time).
  start_time = time.time()
  print("jax.jit compiling...")
  variables = jax.jit(init_model)()
  print("jax.jit compile time: {:.2f}s".format(time.time() - start_time))

  params = variables['params']
  param_count = sum(x.size for x in jax.tree_leaves(params))
  print("param_count: ", param_count)

  total_steps = num_steps
  lr_fn = lambda lr: 0.001

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, accum_steps=accum_steps, lr_fn=lr_fn)

  # Create optimizer and replicate it over all TPUs/GPUs
  opt = momentum_clip.Optimizer(dtype=opt_dtype).create(params)

  initial_step = 1
  opt_repl = flax.jax_utils.replicate(opt)

  # Delete references to the objects that are not needed anymore
  del opt
  del params

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

  # Run training loop
  print('Starting training loop; initial compile can take a while...')
  step_start_time = time.time()
  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, n_prefetch=2)):

    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      opt_repl, loss_repl, update_rng_repl = update_fn_repl(
          opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)

    train_loss = float(flax.jax_utils.unreplicate(loss_repl))

    time_spent = time.time() - step_start_time
    print(
      f'Step: {step}/{total_steps}, '
      f'sec/step: {time_spent:.4f}, '
      f'loss: {train_loss:.4f}'
    )
    step_start_time = time.time()

  return flax.jax_utils.unreplicate(opt_repl)

train()
