# Unmasked: Reconstructing Faces Occluded by Masks using Machine Learning
Grant Perkins, Tian Yu Fan, Jean Claude Zarate, Mingjie Zeng

## Abstract
The accuracy of existing face detection and recognition models are greatly compromised when obstructions, like surgical masks, occlude facial features of the individual. This paper introduces a method which uses facial reconstruction as a new entry for face detection in a mask-wearing context. In the first phase of the model, an object detection network, called EfficientDet-D0, locates the position of the mask. In the second phase, a Gated Convolutional Network and SN-PatchGAN model, in a Generative Adversarial Network, work collaboratively to reconstruct the occluded region of the face. The EfficientDet-D0 model successfully detects masks with a prediction score of 0.966 mAP with an IoU threshold of 50%. The GAN was trained successfully and reconstructed the outline of a human face, but failed to reproduce detailed facial features due to time and hardware constraints.

## Requirements
- Docker
- built base docker image
- nvidia-docker
