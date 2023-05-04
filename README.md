# infant-movements
Automated identification of abnormal infant movements from smart phone videos

The preprint related to this repository can be found https://www.medrxiv.org/content/10.1101/2023.04.03.23288092v1 

Here we present a framework for automating the General Movements Assessment (GMA) using smat phone videos acquired outside the clinical setting in infants between 12 to 18 weeks term corrected age.   

We used Deep Lab Cut to generate a model to track 18 key body points on infants.
In this repository we provide the scripts to pre process the outputs from Deep Lab Cut (pre_processing.py) and the script to train our classifier to predict the GMA from the video movement data cross-validation.py
