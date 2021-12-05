# On Cloud TPU node (use alpha!!!), run
"""
pip install --upgrade pip
export PATH=/home/yfeng_us/.local/bin:${PATH}

pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip uninstall -y six typing-extensions tf-nightly
pip install tensorflow==2.7.0 flax einops tensorflow_datasets

# Clone repository and pull latest changes.
rm -rf vision_transformer || true
git clone --depth=1 https://github.com/yf225/vision_transformer -b vit_dummy_data
cd vision_transformer/

export PYTHONPATH=/home/yfeng_us/vision_transformer:${PYTHONPATH}

python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=tpu --mode=eager --bits=16 --micro-batch-size=8

python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=tpu --mode=graph --bits=16 --micro-batch-size=8

python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=tpu --use_only_one_tpu_core=True --mode=eager --bits=16 --micro-batch-size=1

python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=tpu --use_only_one_tpu_core=True --mode=graph --bits=16 --micro-batch-size=1
"""

# Or, on AWS GPU node, run
"""
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
pip install tensorflow==2.7.0 flax einops tensorflow_datasets tbp-nightly

# Clone repository and pull latest changes.
cd /fsx/users/willfeng/repos
rm -rf vision_transformer || true
git clone --depth=1 https://github.com/yf225/vision_transformer -b vit_dummy_data
cd vision_transformer/

export PYTHONPATH=/fsx/users/willfeng/repos/vision_transformer:${PYTHONPATH}
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PATH=/usr/local/cuda-11.1/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=gpu --mode=eager --bits=16 --micro-batch-size=16

CUDA_VISIBLE_DEVICES=0 python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=gpu --mode=eager --bits=16 --micro-batch-size=16

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 vit_jax/train_vit_jax_tpu_or_gpu_no_dp.py --device=gpu --mode=graph --bits=16 --micro-batch-size=32
"""

# How to view profiler trace on MBP
"""
pip install tensorflow tbp-nightly

rsync -avr ab101835-ddb5-466f-9d25-55b1d5a16351:/fsx/users/willfeng/repos/vision_transformer/tensorboard_trace/* ~/jax_gpu_tensorboard_trace/
tensorboard --logdir=~/jax_gpu_tensorboard_trace
"""

# References:
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax.ipynb
# - https://github.com/google/flax/blob/main/examples/imagenet/train.py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str)
parser.add_argument("--use_only_one_tpu_core", type=bool, default=False)  # only works for TPU
parser.add_argument("--use_only_one_gpu", type=bool, default=False)  # only works for GPU
parser.add_argument("--mode", type=str)
parser.add_argument("--bits", type=int)
parser.add_argument("--micro-batch-size", type=int)
args = parser.parse_args()

assert args.use_only_one_tpu_core or args.use_only_one_gpu
assert args.device in ["tpu", "gpu"]
assert args.mode in ["eager", "graph"]
import jax
import os
if args.device == "tpu":
  # Google Colab "TPU" runtimes are configured in "2VM mode", meaning that JAX
  # cannot see the TPUs because they're not directly attached. Instead we need to
  # setup JAX to communicate with a second machine that has the TPUs attached.
  if 'COLAB_TPU_ADDR' in os.environ:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
  assert "tpu" in str(jax.local_devices()[0]).lower()
  assert jax.local_device_count() == 8
elif args.device == "gpu":
  assert "gpu" in str(jax.local_devices()[0]).lower()
  assert jax.local_device_count() == len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
device = jax.local_devices()[0]
devices = [device]

import functools
import time
import statistics

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
# Prevent TF from messing GPU state
tf.config.experimental.set_visible_devices([], 'GPU')

DEBUG = False
VERBOSE = True
should_profile = True

# Hyperparams
num_attention_heads = 16
hidden_size = 1280
num_layers = 32

micro_batch_size = args.micro_batch_size  # batch size per TPU core or GPU chip

bits = args.bits
assert bits in [16, 32]
if bits == 16:
  if args.device == "tpu":
    model_dtype = jnp.bfloat16
    input_dtype = tf.bfloat16
    opt_dtype = 'bfloat16'
  elif args.device == "gpu":
    model_dtype = jnp.float16
    input_dtype = tf.float16
    opt_dtype = 'float16'
elif bits == 32:
  model_dtype = jnp.float32
  input_dtype = tf.float32
  opt_dtype = 'float32'

if DEBUG:
  print("Overwriting hyperparams since we are in DEBUG mode...")
  num_attention_heads = 1
  hidden_size = 128
  num_layers = 1
  micro_batch_size = 1  # batch size per TPU core or GPU chip

def print_verbose(message):
  if VERBOSE:
    print(message, flush=True)

print("micro_batch_size: {}".format(micro_batch_size))

num_steps = 20
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
    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      logits = apply_fn(
          dict(params=params),
          rngs=dict(dropout=rng),
          inputs=images,
          train=True)
      return cross_entropy_loss(logits=logits, labels=labels)

    l, g = utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), opt.target, batch[0], batch[1],
        accum_steps)

    opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
    return opt, l, rng

  return update_fn


def get_random_data(*, num_classes,
             image_size, global_batch_size, num_steps):
  num_devices = len(devices)

  data = tf.data.Dataset.from_tensor_slices((
    tf.convert_to_tensor(np.random.randn(1, global_batch_size, image_size, image_size, 3) , dtype=input_dtype),
    # tf.one_hot(np.random.randint(0, num_classes, size=(1, global_batch_size, 1)), num_classes),
    tf.one_hot(np.zeros((1, global_batch_size, 1)), num_classes),
  ))

  return data.repeat(num_steps + 1).prefetch(2)


def train():
  """Runs training interleaved with evaluation."""

  # Setup input pipeline
  global_batch_size = micro_batch_size * len(devices)
  ds_train = get_random_data(num_classes=num_classes, image_size=image_size, global_batch_size=global_batch_size, num_steps=num_steps)
  batch = next(iter(ds_train))
  print_verbose((batch[0].shape, batch[1].shape))

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
        jnp.ones(batch[0].shape, model.dtype),
        train=False)

  if args.mode == "eager":
    print_verbose("Skipping jax.jit...")
    variables = init_model()
  elif args.mode == "graph":
    # This compiles the model to XLA (takes some minutes the first time).
    start_time = time.time()
    print_verbose("jax.jit compiling...")
    variables = jax.jit(init_model, backend='cpu')()
    print_verbose("jax.jit compile time: {:.2f}s".format(time.time() - start_time))

  params = jax.device_put(variables['params'], device)
  param_count = sum(x.size for x in jax.tree_leaves(params))
  print_verbose("param_count: {}".format(str(param_count)))

  total_steps = num_steps
  lr_fn = lambda lr: 0.001

  update_fn_obj = make_update_fn(
      apply_fn=model.apply, accum_steps=accum_steps, lr_fn=lr_fn)

  opt = jax.device_put(momentum_clip.Optimizer(dtype=opt_dtype).create(params), device)
  del params

  initial_step = 1

  rng = jax.device_put(jax.random.PRNGKey(0), device)

  # Run training loop
  print_verbose('Starting training loop; initial compile can take a while...')
  step_start_time = time.time()
  step_duration_list = []

  if should_profile:
    jax.profiler.start_trace("./tensorboard_trace")

  for step, batch in zip(
      range(initial_step, total_steps + 1),
      input_pipeline.prefetch(ds_train, None, devices=devices)):

    opt, loss, _ = update_fn_obj(
        opt, step, batch, rng)

    train_loss = loss

    time_spent = time.time() - step_start_time
    step_duration_list.append(time_spent)
    print_verbose(
      f'Step: {step}/{total_steps}, '
      f'sec/step: {time_spent:.4f}, '
      f'loss: {train_loss:.4f}'
    )
    step_start_time = time.time()

  if should_profile:
    jax.profiler.stop_trace()

  print("mode: {}, bits: {}, global_batch_size: {}, micro_batch_size: {}, median time / step: {}".format(args.mode, bits, global_batch_size, micro_batch_size, statistics.median(step_duration_list)))

  return None

train()
