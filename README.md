# Y-Net

This code implements the method that is described in the paper: 

[1] Joris Roels, Julian Hennies, Yvan Saeys, Wilfried Philips, Anna Kreshuk, ["Domain adaptive segmentation in volume electron microscopy imaging"](https://arxiv.org/abs/1810.09734), ISBI, 2019.

## Requirements
- Tested with Python 3.6
- Required Python libraries (these can be installed with `pip install -r requirements.txt`): 
- Data to test the notebook example (optional): 
  - [EPFL mitochondria dataset (source)](https://cvlab.epfl.ch/data/data-em/)
  - [VNC mitochondria dataset (target)](https://github.com/unidesigner/groundtruth-drosophila-vnc)

## Usage
We provide test scripts for Y-Net [1] that illustrates the usage of our method (both unsupervised and semi-supervised domain adaptation). Note that the data paths might be different, depending on where you downloaded the source and target data. The data volumes should be converted to PNG sequences. The [neuralnets](https://github.com/jorisroels/neuralnets) library contains various conversion scripts for volume data. 
