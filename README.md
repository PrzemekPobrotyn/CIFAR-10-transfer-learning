# CIFAR-10 TRANSFER LEARNING

This repository contains a transfer learning exercises on CIFAR-10 done in Keras.

We provided utilities to download, extract and visualise the data.

A number of models are fitted:
* baseline: HOG features + linear SVM
* SVM on top of CNN codes extracted using ResNet50 pretrained on ImageNet
* Fine tuning of ResNet50 (with discussion of suitability of Keras BN layer to fine tuning task)
* Fine tuning with data augmentation

Both development and training were conducted on Google Colab.
If you want to recreate Google Colab environment locally, `pip install -r requirements.txt` in your virtualennv.

In `custom_resnet` directory you can find code needed to perform transfer learning with ResNet50 in Keras. It was adapted from [`keras-applications`](https://github.com/keras-team/keras-applications) by Keras Team and [`keras-resnet`](https://github.com/broadinstitute/keras-resnet) by Broad Institute. 