# PS-000.0-Slidin-videos-Slide-Transition-Detection-and-Title-Extraction-in-Lecture-Video [ITU Challenge 2022]

https://challenge.aiforgood.itu.int/match/matchitem/74

Title: Multi-loss Function to Improve the Text Detection and Segmentation

Team name: AA_Vision
Team members:
1.	Dr. Anuj Abraham, Senior Researcher TII, Abu Dhabi, UAE.
2.	Dr. Shitala Prasad Scientist II, A*STAR, Singapore.

# Overview
In this current task, we need to address two different task detection and segmentation of titles for which we have a joint learning concept. The problem 
statement/motivation can be defined as below: 
YouTube’s “Video Chapter” feature segments a video into sections marked by timestamps so that the user can easily navigate to the part of the video which is of most interest. This can be done by clicking or pressing the chapter marker, or by selecting the timestamp in the video description.

# Environment - Software and Hardware requirements
  •	Linux ubuntu 18.04 LTS
  •	Python - 3.9.13
  •	PyTorch 
  •	Sklearn
  •	Cuda – 11.6
  •	Nvidia GeForce GTX 1080Ti GPU x 2, RAM-11GB

# Parameter setting
  • Trainable parameters: 60996202
  
  • Training time: 8.6 minutes / epoch X 10, Batch size: 4 images, Train model weight size: 235MB
  
  • Training set: 2528, Validation set: 506, 
  
  • Total: 2528 ground truth slides

# Description
  • We used MAE, MSE, MSLE and variations of Huber loss functions for our experiments. These are special ClassRoom loss function from paper: 
  
    Prasad, S., D. Lin, Y. Li, S. Dong, T. L. Nwe., 2021. A Progressive Multi-view Learning Approach for Multi-loss Optimization in 3D Object Recognition. In IEEE Signal Processing Letters, vol. 29, pp. 707-711, 2022, doi: 10.1109/LSP.2021.3132794 
    
  • In our work, we use createDeepLabv3 ResNet 101, as the backbone network.
  
  • In statistical analysis of binary classification, the F-score or F-measure or F1 score is a measure of a test's accuracy. The evaluation is based on F1-score that combines the precision and recall into a single metric by taking their harmonic mean.
  
  • Another metric used is AUROC, wich is the cost difference of accuracy between the ROC curve for the baseline and proposed: Basically, area under the curve.

# Dataset Preparation - Steps involved:
Since this challenge has video as the input source, we needed to extract frames from these video files for feeding deep model. Therefore, for dataset preparation we used following steps: 
1.	The videos frames per second (fps) is 25 
2.	Use video files to extract images, 25 frames per second video
3.	That is, if the video is of X length, the total number of frames will be X(minute)*60(second)*25(fps)
4.	Once the images are extracted, they are categorized into training and validation sets
5.	There are three types of images: no title, same title and new title slides 

# Key contribution and advantages
  • Model used ResNet101 (createDeepLabv3) 
  
  • No architectural changes 
  
  • Negligible model computation cost (equivalent to the original)
  
  • Multi-loss training strategy converges the network much faster
  
  • Gives a significant boost in the performance by ~5% 
  
  # Future work
  In the future work, we would like to further optimize the learning curve with a minimal number of learning parameters.

