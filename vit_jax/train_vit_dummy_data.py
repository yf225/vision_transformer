# Use https://github.com/google-research/vision_transformer/blob/main/vit_jax.ipynb as reference

# Google Colab "TPU" runtimes are configured in "2VM mode", meaning that JAX
# cannot see the TPUs because they're not directly attached. Instead we need to
# setup JAX to communicate with a second machine that has the TPUs attached.
import os
if 'COLAB_TPU_ADDR' in os.environ:
  import jax
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()
  print('Connected to TPU.')
  print("jax.device_count(): ", jax.device_count())
  print("jax.local_devices(): ", jax.local_devices())
else:
  print('No TPU detected. Can be changed under "Runtime/Change runtime type".')

# ======

!pip install flax
import functools
import os
import time

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# ======

# Clone repository and pull latest changes.
![ -d vision_transformer ] || git clone --depth=1 https://github.com/google-research/vision_transformer
!cd vision_transformer && git pull

import sys
if './vision_transformer' not in sys.path:
  sys.path.append('./vision_transformer')

%load_ext autoreload
%autoreload 2

from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip
from vit_jax import utils

# ======

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

micro_batch_size = 28  # batch size per TPU core
global_batch_size = micro_batch_size * jax.device_count()

num_steps = 4
learning_rate = 0.001

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
num_classes = 1000
dropout_rate = 0.

# ======

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
        jax.value_and_grad(loss_fn), opt.target, batch['image'], batch['label'],
        accum_steps)
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
    l = jax.lax.pmean(l, axis_name='batch')

    opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
    return opt, l, new_rng

  return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))


def get_random_data(*, num_classes,
             image_size, num_examples):
  data = tf.data.Dataset.from_tensor_slices((
    np.random.randn(num_examples, (image_size, image_size, 3)),
    tf.one_hot(np.random.randint(0, num_classes, size=(1,)).item(), num_classes)
  ))

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, image_size, image_size, 3])
    data['label'] = tf.reshape(data['label'],
                               [num_devices, -1, num_classes])
    return data

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)


def train():
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  num_examples = global_batch_size * num_steps
  ds_train = get_random_data(num_classes=num_classes, image_size=image_size, num_examples=num_examples)
  batch = next(iter(ds_train))

  # Build VisionTransformer architecture
  model = models.VisionTransformer(
    num_heads=num_attention_heads,
    hidden_size=hidden_size,
    num_layers=num_layers,
    patch_size=patch_size,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
  )

  def init_model():
    return model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension for initialization.
        jnp.ones(batch['image'].shape[1:], batch['image'].dtype.name),
        train=False)

  # Use JIT to make sure params reside in CPU memory.
  variables = jax.jit(init_model, backend='cpu')()

  params = variables['params']

  total_steps = num_steps
  lr_fn = lambda lr: 0.001

  update_fn_repl = make_update_fn(
      apply_fn=model.apply, accum_steps=1, lr_fn=lr_fn)

  # Create optimizer and replicate it over all TPUs/GPUs
  opt = momentum_clip.Optimizer(dtype='bfloat16').create(params)

  initial_step = 1
  opt_repl = flax.jax_utils.replicate(opt)

  # Delete references to the objects that are not needed anymore
  del opt
  del params

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

  # Run training loop
  logging.info('Starting training loop; initial compile can take a while...')
  t0 = lt0 = time.time()
  lstep = initial_step
  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, 1)):

    with jax.profiler.StepTraceContext('train', step_num=step):
      opt_repl, loss_repl, update_rng_repl = update_fn_repl(
          opt_repl, flax.jax_utils.replicate(step), batch, update_rng_repl)

    if step == initial_step:
      logging.info('First step took %.1f seconds.', time.time() - t0)
      t0 = time.time()
      lt0, lstep = time.time(), step
    else:
      # Report training metrics per step
      time_spent = time.time() - lt0
      lt0, lstep = time.time(), step
      done = step / total_steps
      logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '  # pylint: disable=logging-format-interpolation
                    f'sec/step: {time_spent:.2f}, '
                    f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

  return flax.jax_utils.unreplicate(opt_repl)

train()
