+++
date = "20 April 2018"
draft = true
title = "Class 12: Intellectual Property Protection"
author = "Team Panda"
slug = "class12"
+++

## DeepSigns: A Generic Watermarking Framework for IP Protection of Deep Learning Models

> Bita Darvish Rohani, Huili Chen, Farinaz Koushanfar. _DeepSigns: A Generic Watermarking Framework for IP Protection of Deep Learning Models._ April 2018. arXiv e-print [[PDF]](https://arxiv.org/pdf/1804.00750.pdf) 

As Deep Learning technology advances, it transitions from solely a research topic to a useful tool with several real world applications.  Because of the increasing commercialization of Deep Learning models, there is a growing interest in protecting models, as they have become a valuable piece of intellectual property.  It is well-known that training accurate DL models is computationally expensive, which serves as a strong motivation for companies to prevent competitors from stealing their final model without incurring any of the training costs.  One solution to this problem is to add a digital watermark to the model, much like the watermarks that appear on copyrighted photos and videos.  This paper proposes one such method to add a watermark to a given model, so that if the model were to be taken and used without permission, it could be proven to be the original owner's intellectual property.  

Previous work [[1]](https://arxiv.org/pdf/1701.04082.pdf) has added watermarks to convolutional neural networks, but were notably data and model specific - a problem for encouraging the widespread use of such techniques.  Other methods have also been proposed [[2]](https://arxiv.org/pdf/1711.01894.pdf), but are only applicable to models with black-box models and the watermarks are too vulnerable to manipulation to serve as reliable proof of ownership.  In response, this paper proposes a method called DeepSigns, a generic method for watermarking models that can be applied to both white-box and black-box models, while being robust against several removal attacks.

### Aims

A watermarking process must have several traits if it is to be effective in protecting IP while maintaining the model's usefulness.  Specifically, this paper describes a watermarking process as requiring the following characteristics: fidelity, capacity, efficiency, security, robustness, reliability, integrity, and generalizability.  In brief, a watermark needs to maintain model accuracy, be generated quickly, be robust to attacks, yield minimal false negatives and false positives, and be applicable to both white-box and black-box scenarios.  With these goals in mind, the authors describe their watermarking method, DeepSigns. 

### Watermark Generation

In this framework, the hidden layers and output layer are treated separately, since the hidden layers have continuous outputs to the next layers while the output layer returns a discrete label.  The hidden layers of the network are assumed to have a Gaussian Mixture Model (GMM) prior data distribution, so the authors add an additional loss function to the standard cross-entropy loss for training DL models to account for this.  

<p align="center">
<img src="/images/class12/equation1.png" width="400">
</p> 

This allows for the underlying probability density function for each layer to be more accurately estimated without interference from previous layers.  To actually create the watermark, the model owner first generates three things: (1) a set of random indices, each corresponding to a single Gaussian distribution in the model, (2) an arbitrary, evenly distributed binary string which is to be embedded in the model, and (3) a random projection matrix to map the mean values from the chosen Gaussian distributions into the selected binary string.  

<p align="center">
<img src="/images/class12/equation2.png" width="400">
</p> 

To incorporate this matrix into the training process, a third loss function is added, so that the model is now trained using all three loss functions simultaneously. 

<p align="center">
<img src="/images/class12/equation3.png" width="400">
</p> 

To watermark the output layer, one must be very careful so as to not change the final prediction vectors and ruin the model's overall accuracy.  First, the probability density function of the intermediate layers must be known, which is achieved through watermarking the hidden layers as described above.  Next, k random input samples are chosen to be watermarking keys.  These samples are chosen such that their features lie within the unused regions of the model and are independent from other potential nearby samples.  The chosen k inputs are randomly assigned prediction vectors, and then the pre-trained model is trained again, this time with the set of watermarking keys.  In this way, the model learns to misclassify the keys from their ground truth classes to the randomly assigned classes, without skewing the outputs for other samples because the keys were selected specifically for their independence from other inputs.  

### Watermark Extraction

To extract the watermark, the owner would simply send a set of inputs, including the key inputs, to query the model.  For a black-box model, the owner would compare the outputs for the keys with the known outputs that were randomly assigned to the key inputs.  Since those are not the correct ground truth classifications for those samples, if more than a certain threshold match the random assignment, it is statistically likely that the model was stolen, allowing the original creator to claim ownership.  For a white-box model, the above process could be performed, as well as performing inverse operations on the intermediate layers to retrieve the embedded binary string.

### Evaluation

The performance of DeepSigns was evaluated on two different datasets (CIFAR-10 and MNIST), three different neural network architectures (MLP, CNN, and WideResNet), and against three potential attacks (parameter pruning, model fine-tuning, and watermark overwriting).  In this context, model pruning would be an attack that obscured the watermark by compressing the network, model fine-tuning would be an attack that re-trained the model, altering the final prediction vectors, and watermark overwriting would be an attack that, as the name suggests, inserts a new watermark as a way to hide the previous one.  

<p align="center">
<img src="/images/class12/evaluation_chart.png" width="650">
</p>

As can be seen below, all three attacks were successful in eventually obscuring the watermark, but by the time they achieved this, the model's accuracy had been significantly decreased.  The authors claim this decrease in accuracy to be demonstrative of DeepSigns' robustness to these attacks, but a slight drop in accuracy may be permissible when the training cost was free to the party that stole the model.  This paper demonstrates progress towards the protection of DL models as intellectual property, but it seems more work will need to be done before a company could reliably claim a model as their own and cite the watermark as proof of ownership in court.

<p align="center">
<img src="/images/class12/pruning_attack1.png" width="500">
<img src="/images/class12/pruning_attack2.png" width="500">
</p> 
<p align="center">
<img src="/images/class12/fine_tuning_attack.png" width="700">
</p> 
<p align="center">
<img src="/images/class12/watermark_overwriting.png" width="600">
</p> 

### References

[[1]](https://arxiv.org/pdf/1701.04082.pdf) Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, “Embedding watermarks into deep neural networks,” in Proceedings of the 2017 ACM on International Conference on Multimedia Retrieval. ACM, 2017, pp. 269–277.

[[2]](https://arxiv.org/pdf/1711.01894.pdf) E. L. Merrer, P. Perez, and G. Tredan, “Adversarial frontier stitching for remote neural network watermarking,” arXiv preprint arXiv:1711.01894, 2017.

## Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring

> Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. _Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring_ February 2018. arXiv e-print [[PDF]](https://arxiv.org/pdf/1802.04633.pdf) 

### Introduction
As deep learning models are more readily adopted in industry, the task of protecting models from being stolen will become a more pressing issue. The costs associated with collecting training data and training a model are high and models can be easily copied. Inorder to identify stolen models, this paper introduces watermarking for deep learning networks that has no major impact on the performance of the model.

Watermarking is not a new technology and is more commonly used in video and audio media to identify stolen intellectual property. The authors found that present watermarking methods for deep learning lacked security properties or operated on the outputs of the model instead of the model itself.

### Backdooring Definition

The goal of backdooring is to get the model to output "wrong" labels \\( T_L \\) on certain inputs \\( T \\).  \\( T \\) is a subset of the inputs and is called the _trigger set_. These triggers give outputs that we assign. The goal of the backdooring algorithm is to misclassify the trigger images with high probability. The backdoor can be applied to a pre-trained model or a applied to a model during training.

The authors describe strong backdoors as being those which are hard to remove without access to the trigger set. They also require that strong back doors return a minimal trigger set size and two backdoor trigger sets don’t intersect.

### Watermarking Properties
The authors describe four properties a watermarking algorithm should follow.

* Preserve functionality
* Unremovable
* Unforgeable
* Enforce non-trivial ownership

The model with a watermark must have the same or close to the same accuracy of the model without a watermark. The watermark should not be able to be removed by an adversary even if the adversary knows that the model uses a watermark and how the watermark was generated. Additionally, the adversary shouldn’t be able to forge answers to the algorithm that verifies the watermarked model. This means that the adversary can’t claim ownership over a model.


### Implementation and Experiments

This paper describes two approaches for generating watermarks in models. The PreTrained approach applies the watermarks after a model has been trained and the FromScratch approach applies the watermark during training. This paper used CIFAR-10, CIFAR-100, and ImageNet image classification datasets to demonstrate the properties of their method mentioned in the Watermarking Properties section.

<p align="center">
<img src="/images/class12/back_figure5.PNG" width="400">
<br> <b>Figure:</b> Example trigger set image; the label for this image was “automobile.”    
</p>

The figure above is an example of a trigger set image, which is abstract and is given a random label. The trigger set images should be uncorrelated so that revealing a subset of the trigger set images should not represent other images in the set. This is especially important for public verifiability when a subset of the trigger set must be provided. 

<p align="center">
<img src="/images/class12/back_table1.PNG" width="400">
<br> <b>Table:</b> Classification accuracies for models with and without a watermark.
</p>

The above table shows that the model trained normally (No-WM) has no significant advantage over the watermarked models. Additionally the watermarked models were perfectly accurate on the trigger set images. 

#### Fine-Tuning

This paper also covers preservation of a water mark after fine tuning. Fine tuning is training different layers of the models at different rates so that common model features are preserved for different model applications. The different methods of fine-tuning are mentioned below. Re-training initializes certain layers with random weights while fine-tuning updates the weights on the specified layers which are initizalized from training previously.

* Fine-Tune Last Layer (FTLL)
* Fine-Tune All Layers (FTAL)
* Re-Train Last Layers (RTLL)
* Re-Train All Layer (RTAL)

The results for the different fine tuning methods are given below. For problems with more classes such as CIFAR-100, it's clear that using the FromScratch approach makes a major difference in the preservation trigger set accuracy.

<p align="center">
<img src="/images/class12/back_figure6.PNG" width="400">
<br> <b>Figure:</b> Classification accuracies for models with and without a watermark.
</p>

### Conclusion

This paper introduces approaches to applying watermarking methods to deep learning models. By using the overparameterization of deep learning models, answers to randomly assigned trigger set images can be stored in models without impacting the usefulness of the model. This watermarking technique can be used to identify stolen models and preserve intellectual property.
