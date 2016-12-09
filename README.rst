.. -*- mode: rst -*-

ND DIAL: Imbalanced Algorithms
==============================

Algorithms for imbalanced learning, with a focus on using deep/representational learning. Below is a list of the methods currently implemented.

* Oversampling
    1. SMOTE - Synthetic Minority Over-sampling Technique [1]_
    2. DAE - Denoising Autoencoder [2]_
    3. VAE - Variational Autoencoder [3]_

* Ensemble Sampling
    1. SMOTEBoost [4]_

References:
-----------

.. [1] : N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE: Synthetic Minority Over-Sampling Technique." Journal of Artificial Intelligence Research (JAIR), 2002.

.. [2] : P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol. "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion". Journal of Machine Learning Research (JMLR), 2010.

.. [3] : D. P. Kingma and M. Welling. "Auto-Encoding Variational Bayes". arXiv preprint arXiv:1312.6114, 2013.

.. [4] : N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer. "SMOTEBoost: Improving Prediction of the Minority Class in Boosting." European Conference on Principles of Data Mining and Knowledge Discovery (PKDD), 2003.