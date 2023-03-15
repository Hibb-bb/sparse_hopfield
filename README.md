# Sparse Hopfield Networks 

---

## Requirements

The software was developed and tested on the following 64-bit operating systems:

- CentOS Linux release 8.1.1911 (Core)
- macOS 10.15.5 (Catalina)

As the development environment, [Python](https://www.python.org) 3.8.3 in combination
with [PyTorch](https://pytorch.org) 1.6.0 was used (a version of at least 1.5.0 should be sufficient). More details on
how to install PyTorch are available on the [official project page](https://pytorch.org).

## Installation

The recommended way to install the software is to use `pip/pip3`:


```bash
$ pip3 install -r examples/requirements.txt
```

## DeepRC Dataset Usage

**Please do not use the cmv_implanted_dataset yet, the filelink now is missing.**
**THe simulated dataset is still under testing**
**To download the lstm_generated and CMV dataset, please run `python3 download_dataset.py`**

```python
from deeprc.dataset_readers import make_dataloaders
from deeprc.predefined_datasets import *

batch_size = 4

# For the CMV dataset
task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_dataset(dataset_path='./datasets/cmv/', batch_size=batch_size)

# For the CMV + Implant Signal  dataset

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_implanted_dataset(dataset_path='./datasets/cmv_implanted/', batch_size=batch_size)

# For the Simulated dataset

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = simulated_dataset(dataset_path='./datasets/simulated/', batch_size=batch_size)

# For the lstm generated dataset

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = lstm_generated_dataset(dataset_path='./datasets/lstm/', batch_size=batch_size)

```

## Usage

To get up and running with Hopfield-based networks, only <i>one</i> argument needs to be set, the size (depth) of the
input.

```python
from hflayers import Hopfield

hopfield = Hopfield(input_size=...)
```

It is also possible to replace commonly used pooling functions with a Hopfield-based one. Internally, a <i>state
pattern</i> is trained, which in turn is used to compute pooling weights with respect to the input.

```python
from hflayers import HopfieldPooling

hopfield_pooling = HopfieldPooling(input_size=...)
```

A second variant of our Hopfield-based modules is one which employs a trainable but fixed lookup mechanism. Internally,
one or multiple <i>stored patterns</i> and <i>pattern projections</i> are trained (optionally in a non-shared manner),
which in turn are used as a lookup mechanism independent of the input data.

```python
from hflayers import HopfieldLayer

hopfield_lookup = HopfieldLayer(input_size=...)
```

The usage is as <i>simple</i> as with the main module, but equally <i>powerful</i>.

## Examples

Generally, the Hopfield layer is designed to be used to implement or to substitute different layers like:

- <b>Pooling layers:</b> We consider the Hopfield layer as a pooling layer if only one static state (query) pattern
  exists. Then, it is de facto a pooling over the sequence, which results from the softmax values applied on the stored
  patterns. Therefore, our Hopfield layer can act as a pooling layer.

- <b>Permutation equivariant layers:</b> Our Hopfield layer can be used as a plug-in replacement for permutation
  equivariant layers. Since the Hopfield layer is an associative memory it assumes no dependency between the input
  patterns.

- <b>GRU & LSTM layers:</b> Our Hopfield layer can be used as a plug-in replacement for GRU & LSTM layers. Optionally,
  for substituting GRU & LSTM layers, positional encoding might be considered.

- <b>Attention layers:</b>  Our Hopfield layer can act as an attention layer, where state (query) and stored (key)
  patterns are different, and need to be associated.

The folder [examples](examples/) contains multiple demonstrations on how to use the <code>Hopfield</code>, <code>
HopfieldPooling</code> as well as the <code>HopfieldLayer</code> modules. To successfully run the
contained [Jupyter notebooks](https://jupyter.org), additional third-party modules
like [pandas](https://pandas.pydata.org) and [seaborn](https://seaborn.pydata.org) are required.

- [Bit Pattern Set](examples/bit_pattern/bit_pattern_demo.ipynb): The dataset of this demonstration falls into the
  category of <i>binary classification</i> tasks in the domain of <i>Multiple Instance Learning (MIL)</i> problems. Each
  bag comprises a collection of bit pattern instances, wheres each instance is a sequence of <b>0s</b> and <b>1s</b>.
  The positive class has specific bit patterns injected, which are absent in the negative one. This demonstration shows,
  that <code>Hopfield</code>, <code>HopfieldPooling</code> and <code>HopfieldLayer</code> are capable of learning and
  filtering each bag with respect to the class-defining bit patterns.

- [Latch Sequence Set](examples/latch_sequence/latch_sequence_demo.ipynb): We study an easy example of learning
  long-term dependencies by using a simple <i>latch task</i>,
  see [Hochreiter and Mozer](https://link.springer.com/chapter/10.1007/3-540-44668-0_92). The essence of this task is
  that a sequence of inputs is presented, beginning with one of two symbols, <b>A</b> or <b>B</b>, and after a variable
  number of time steps, the model has to output a corresponding symbol. Thus, the task requires memorizing the original
  input over time. It has to be noted, that both class-defining symbols must only appear at the first position of a
  sequence. This task was specifically designed to demonstrate the capability of recurrent neural networks to capture
  long term dependencies. This demonstration shows, that <code>Hopfield</code>, <code>HopfieldPooling</code> and <code>
  HopfieldLayer</code> adapt extremely fast to this specific task, concentrating only on the first entry of the
  sequence.

- [Attention-based Deep Multiple Instance Learning](examples/mnist_bags/mnist_bags_demo.ipynb): The dataset of this
  demonstration falls into the category of <i>binary classification</i> tasks in the domain of <i>Multiple Instance
  Learning (MIL)</i> problems, see [Ilse and Tomczak](https://arxiv.org/abs/1802.04712). Each bag comprises a collection
  of <b>28x28</b> grayscale images/instances, whereas each instance is a sequence of pixel values in the range
  of <b>[0; 255]</b>. The amount of instances per pag is drawn from a Gaussian with specified mean and variance. The
  positive class is defined by the presence of the target number/digit, whereas the negative one by its absence.

## Disclaimer

Some implementations of this repository are based on existing ones of the
official [PyTorch repository v1.6.0](https://github.com/pytorch/pytorch/tree/v1.6.0) and accordingly extended and
modified. In the following, the involved parts are listed:

- The implementation of [HopfieldCore](hflayers/activation.py#L16) is based on the implementation
  of [MultiheadAttention](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/activation.py#L771)
  .
- The implementation of [hopfield_core_forward](hflayers/functional.py#L8) is based on the implementation
  of [multi_head_attention_forward](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/functional.py#L3854)
  .
- The implementation of [HopfieldEncoderLayer](hflayers/transformer.py#L12) is based on the implementation
  of [TransformerEncoderLayer](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/transformer.py#L241)
  .
- The implementation of [HopfieldDecoderLayer](hflayers/transformer.py#L101) is based on the implementation
  of [TransformerDecoderLayer](https://github.com/pytorch/pytorch/blob/b31f58de6fa8bbda5353b3c77d9be4914399724d/torch/nn/modules/transformer.py#L303)
  .

## License

This repository is BSD-style licensed (see [LICENSE](LICENSE)), except where noted otherwise.
