## Recurrent Neural Network for Blind Image Quality Assessment, version 2

This is an improved version of the RNN-BIQA, https://github.com/jarikorhonen/rnnbiqa. The main difference is that the low and high resolution patches are processed in separate streams. The pipeline for generating features and training the model is similar, with some minor differences.

As a prerequisite, the following third-party image quality databases need to be installed:

LIVE Challenge image quality database from: http://live.ece.utexas.edu/research/ChallengeDB/ (for training the featrue extractor)

KoNIQ-10k image quality database from: http://database.mmsp-kn.de/koniq-10k-database.html (for training and testing the RNN model)

SPAQ image quality database from: https://github.com/h4nwei/SPAQ (for training and testing the RNN model)

In addition, the additional low and high quality patches for training the feature extractor are available for download [here](https://mega.nz/file/vVYwgKAI#9PXT-KmnWTlRbUvdd4cgTTjoEo6fmGIUJvvZhrAE2Tc). The ZIP file should be extracted under the same directory where the generated LIVE Challenge patches are stored (by default, the directory is _.\\livec_patches_, and therefore the additional patches should be in directories _.\\livec_patches\\hq_images_ and _.\\livec_patches\\hq_images_.

For using the implementation, download all the Matlab scripts in the same folder.

For training and testing the model from scratch, you can use `masterScript.m`. It can be run from 
Matlab command line as:

```
>> masterScript(livec_path, koniq_path, spaq_path, cpugpu);
```

The following input is required:

`livec_path`: path to the LIVE Challenge dataset, including metadata files _allmos_release.mat_ and 
_allstddev_release.mat_. For example: _'c:\\livechallenge'_.

`koniq_path`: path to the KoNIQ-10k dataset, including metadata file 
_koniq10k_scores_and_distributions.csv_. For example: _'c:\\koniq10k'_.

`spaq_path`: path to the SPAQ dataset, including metadata _file mos_spaq.xlsx_. For example: 
_'c:\\spaq'_.

`cpugpu`: whether to use CPU or GPU for training and testing the models, either _'cpu'_ or _'gpu'_.

The script implements the following functionality:

1) Makes patches out of LIVE Challenge dataset and makes probabilistic quality scores (file 
_LiveC_prob.mat_), `using processLiveChallenge.m` script.
2) Makes downscaled version of the SPAQ dataset (SPAQ-768), using `resizeImages.m` script.
3) Trains CNN feature extractor, using `trainCNNmodelV2.m` script.
4) Extracts feature vector sequences from KoNIQ-10k and SPAQ images, using the trained
feature extractor and `computeCNNfeaturesV2.m` script.
5) Trains and tests RNN model by using KoNIQ-10k features for training and SPAQ for testing,
and then vice versa. Uses `trainAndTestRNNmodelv2.m` script for this purpose. Displays the results
for SCC, PCC, and RMSE.

You may skip steps 1 and 3 and use the pretrained feature extractor model available to download [here](https://mega.nz/file/qA4GzQ4S#SUC2zdpNnCNvEhYHsV6lewJLslTnrULmEGKm0Iz2VZk). 
