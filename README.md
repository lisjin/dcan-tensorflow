# dcan-tensorflow

## Objective
* Detect boundaries of cell nuclei in images of human U2OS cells using [Deep Contour-Aware Networks](https://arxiv.org/pdf/1604.02677.pdf) (DCAN) model created by Chen et. al
	* Note: in contrast to paper, there is no auxiliary supervision in the upsampled layers
* Image set: https://data.broadinstitute.org/bbbc/BBBC006/

## Results

### Model Output
![iter-12650](/../screenshots/iter-12650.png)
> After 12,650 iterations, the output of the model is on the rightmost column (labels in the middle column).

### TensorBoard Plots
![dice-coefs.png](/../screenshots/dice-coefs.png)
> As shown in the dice coefficient plots above, the model continues to improve beyond the results shown above. Left plot is for contours, right for segments.

## Instructions

All code within `tf-dcan/` was written for the Broad Institute human U2OS cells image set referenced above. This image set contains 384 images at 32 different _z_-indices (from an automated microscope), and are of size 692 x 520.

For running on your own images, you must:
1. Generate a set of ground truth _contours_ and _segments_ as described in the DCAN paper.
	1. Use `tf-dcan/traceBounds.m` as a starting point.
2. Customize `bbbc006.py` and `bbbc006_input.py` in `dcan-tensorflow/` according to your data (e.g., image dimensions, file type decoding).

To run the code on the U2OS image set, follow all steps in the sections below.

### Preprocessing

The `fm-prep/` directory contains MATLAB scripts to automatically select the optimal focal plane for each image.

1. Move all directories from the U2OS image set to the repo's root.
2. Run `fmeasureAll()` in MATLAB to compute each the focus measure of each image.
	1. Make sure to [install](https://www.mathworks.com/matlabcentral/fileexchange/27314-focus-measure?focused=8113992&tab=function&requestedDomain=www.mathworks.com) the `fmeasure` function and change line 10 of `fmeasureAll.m` to point to the directory.
3. Run `saveImgsAll()` in MATLAB to save the detected in-focus images to an output directory called `BBBC006_v1_focused/`.
	1. Change line 6 of `saveImgs.m` if you want to rename the output directory.

### Training

The `tf-dcan/` contains all TensorFlow code, which uses the CIFAR-10 tutorial [code](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10) as a skeleton.

1. Using [virtualenv](https://virtualenv.pypa.io/en/stable/installation/), run the following to install all dependencies.

```bash
virtualenv venv --distribute
source venv/bin/activate
pip install -r requirements.txt
```

2. Run `bbbc006_train.py` (or `bbbc006_multi_gpu_train.py` for multiple GPUs) to train the network.
	1. Adjust the parameters lines 54-60 of `bbbc006.py` to fit your training data (especially lines 59-60, which deal with class imbalance).
3. Run `bbbc006_eval.py` for evaluation using dice coefficient.
4. Use TensorBoard to visualize the results in `/tmp/bbbc006_train` and/or `/tmp/bbbc006_eval` for training and evaluation results, respectively.
