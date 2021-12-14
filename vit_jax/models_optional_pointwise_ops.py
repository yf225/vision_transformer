# Copyright 2021 Google LLC.
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

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


use_bias = False
use_norm = False
use_gelu = False
use_dropout = False


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.bfloat16
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.zeros
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=use_bias)(  # pytype: disable=wrong-arg-types
            inputs)
    if use_gelu:
      x = nn.gelu(x)
    if use_dropout:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=use_bias)(  # pytype: disable=wrong-arg-types
            x)
    if use_dropout:
      output = nn.Dropout(
          rate=self.dropout_rate)(
              output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: bfloat16).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.bfloat16
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    if use_norm:
      x = nn.LayerNorm(dtype=self.dtype)(inputs)
    else:
      x = inputs
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.zeros,
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        use_bias=use_bias)(
            x, x)
    if use_dropout:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    if use_norm:
      y = nn.LayerNorm(dtype=self.dtype)(x)
    else:
      y = x
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    hidden_size: int
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  hidden_size: int
  num_heads: int
  dtype: Dtype = jnp.bfloat16
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, patch_size * patch_size * num_channels)
    num_patches = inputs.shape[1]

    x = nn.Dense(
        features=self.hidden_size,
        name='projection',
        dtype=self.dtype,
        use_bias=use_bias)(inputs)
    x = x + nn.Embed(num_embeddings=num_patches, features=self.hidden_size, dtype=self.dtype)(
      np.arange(start=0, stop=num_patches, step=1)
    )

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=4 * self.hidden_size,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads,
          dtype=self.dtype)(
              x, deterministic=not train)
    if use_norm:
      encoded = nn.LayerNorm(name='encoder_norm', dtype=self.dtype)(x)
    else:
      encoded = x

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_heads: int
  hidden_size: int
  num_layers: int
  patch_size: int
  num_classes: int
  dropout_rate: float
  dtype: Dtype = jnp.bfloat16
  representation_size: Optional[int] = None
  classifier: str = 'token'
  input_is_4D: bool = False

  @nn.compact
  def __call__(self, inputs, *, train):
    x = inputs

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, (h // self.patch_size) * (w // self.patch_size), self.patch_size * self.patch_size * c])

    x = Encoder(
      name='Transformer',
      num_layers=self.num_layers,
      hidden_size=self.hidden_size,
      num_heads=self.num_heads,
      dropout_rate=self.dropout_rate,
      attention_dropout_rate=self.dropout_rate,
      dtype=self.dtype)(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.num_classes:
      x = nn.Dense(
        features=self.num_classes,
        name='head',
        kernel_init=nn.initializers.zeros,
        dtype=self.dtype,
        use_bias=use_bias)(x)
    return x
