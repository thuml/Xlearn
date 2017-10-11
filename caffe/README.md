# Xlearn on Caffe

This is a caffe repository for transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks".
- Add `jmmd layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks".
- Add `entropy layer` and `outerproduct layer` described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks".
- Copy `grl layer` and `messenger.hpp` from repository [Caffe](https://github.com/ddtm/caffe/tree/grl).
- Emit `SOLVER_ITER_CHANGE` message in `solver.cpp` when `iter_` changes.

Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).

Training Model
---------------

In `models/DAN/alexnet`, we give an example model based on Alexnet to show how to transfer from `webcam` to `amazon`. In this model, we insert mmd layers after fc7 and fc8 individually.


In `models/RTN/alexnet`, we give an example model based on Alexnet to show how to transfer from `webcam` to `amazon`. In this model, we insert mmd layers after the outer product of the output of fc7 and fc8.

In `models/JAN/alexnet` and `models/JAN/resnet`, we give an example model based on Alexnet and ResNet respectively to show how to transfer from `webcam` to `amazon`. In this model, we insert jmmd layers with output of fc7 and fc8 as its input.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model for Alexnet. The [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) is used as the pre-trained model for Resnet. We use Resnet-50. If the Office dataset and pre-trained caffemodel are prepared, the example can be run with the following command:
```
Alexnet:

"./build/tools/caffe train -solver models/*/alexnet/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel (*=DAN, RTN or JAN)"
```
```
ResNet:

"./build/tools/caffe train -solver models/JAN/resnet/solver.prototxt -weights models/deep-residual-networks/ResNet-50-model.caffemodel"
```

### Since finetuning ResNet on caffe needs too much memory for one gpu to hold, we don't finetune ResNet. 

Parameter Tuning
---------------
In mmd-layer and jmmd-layer, parameter `loss_weight` can be tuned to give mmd/jmmd loss different weights.

Task Change
---------------
If you want to change the transfer task like change to `amazon` to `dslr`. You need to modify the corresponding `train_val.prototxt` to change the source and target dataset. And you need to change the `test_iter` parameter corresponding `solver.prototxt` to the size of the target dataset: `2817` for `amazon`, `795` for `webcam` and `498` for `dslr`. 
