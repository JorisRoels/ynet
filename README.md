# Y-Net

This code implements the methods that were described in the papers: 

[1] Joris Roels, Julian Hennies, Yvan Saeys, Wilfried Philips, Anna Kreshuk, ["Domain adaptive segmentation in volume electron microscopy imaging"](https://arxiv.org/abs/1810.09734), ISBI, 2019. 

[2] Joris Roels, Anna Kreshuk, Yvan Saeys, ["A Domain Adaptive Segmentation Algorithm for Electron Microscopy Imaging"](https://arxiv.org/abs/1810.09734), Transactions on Medical Imaging (submitted), 2019. 

## Requirements
- Tested with Python 3.6
- Required Python libraries (these can be installed with `pip install -r requirements.txt`): 
    - numpy
    - tifffile
    - scipy
    - scikit-image
    - imgaug
    - torch
    - torchvision
    - h5py
    - jupyter (optional)
    - progressbar2 (optional)
    - tensorboardX (optional, for tensorboard usage)
    - tensorflow (optional, for tensorboard usage)

## Requirements
- Required data: 
  - [EPFL mitochondria dataset (source)](https://cvlab.epfl.ch/data/data-em/)
  - [VNC mitochondria dataset (target)](https://github.com/unidesigner/groundtruth-drosophila-vnc)

## Usage
We provide notebooks for [Y-Net](ynet.ipynb) [1] and [Y-Net-DWS](ynet_dws.ipynb) [2] that illustrate the usage of our methods. Note that the data path might be different, depending on where you downloaded the data. 
