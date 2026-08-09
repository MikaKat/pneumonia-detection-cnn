# Attribution for the demonstration images in this folder

The six radiographs in this folder are redistributed under the terms of use of
the RSNA Pneumonia Detection Challenge, which permit use for research,
education and other purposes provided the sources are named and no attempt is
made to identify the individuals depicted.

## The radiographs

Provided by the **NIH Clinical Center**, from the NIH Chest X-ray collection.

Download site: https://nihcc.app.box.com/v/ChestXray-NIHCC

Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale
Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
Localization of Common Thorax Diseases.* IEEE CVPR 2017.

## The labels and the boxes

From the **RSNA Pneumonia Detection Challenge**, organised by the Radiological
Society of North America with the Society of Thoracic Radiology.

Download site: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge

Shih G, Wu CC, Halabi SS, et al. *Augmenting the National Institutes of Health
Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.*
Radiology: Artificial Intelligence 2019;1(1):e180041.

## Which six, and why these

All six are from the held-out split of 3,812 images. No training run and no
model selection ever saw them.

They were picked by a rule written down before any candidate was looked at: one
image per RSNA class (Normal, No Lung Opacity / Not Normal, Lung Opacity) per
projection (AP, PA), each the one closest to the median ensemble probability of
its cell, ties broken by the smallest identifier, and no swaps afterwards. The
script is `rsna/befunde/rsna_demobilder.py`, the result is `manifest.json` next
to this file, and the reasoning is in
`erklaerungen/31_webapp_karten_und_skala.md`, section 4.
