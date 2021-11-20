# Smartphone-based-RTI
### Geometric and 3D Computer Vision 2021

See <b>[G3DCV2021_FinalProject.pdf](G3DCV2021_FinalProject.pdf) for more details.  

## Introduction
Reflectance Transformation Imaging (RTI) is a technique to capture the reflectance of an
object under different lighting conditions. It enables the interactive relighting of the subject
from any direction and the mathematical enhancement of subject’s surface shape and color
attributes. Typically, RTI uses a static camera and an array of light sources at known (or
knowable) positions.  

## Assignment
In this project, you’ll explore RTI using a cheaper setup composed by two consumer
smartphones equipped with a camera and LED flashlight. The idea is to use one of the
two smartphones as a movable light source while the other captures the object from a
frontal position. We exploit the camera of the movable light source to recover the light
direction vector at each point and provide the same data obtainable with a light dome.

## Requirements
The project must contain 3 different programs:
1. A <code>camera_calibrator</code> that loads one of the two provided calibration videos and
computes the intrinsic parameters of the camera (the intrinsic matrix K and the lens
distortion vector assuming a 5-parameters model) without any user intervention.
2. The <code>analysis</code> program to process a video sequence and compute a texture map of
the object (ie a function, for each object pixel, mapping light direction to pixel
intensity). The whole analysis must be performed without any user intervention.
Tunable parameters should not depend on the specific sequence processed.
3. An <code>interactive_relighting</code> program to render a previously processed object according
to a user-definable light source direction

## Assets
Download the assets folder from [here](http://www.dsi.unive.it/~bergamasco/teachingfiles/G3DCV2021_data.7z).  
Create the <code>assets/</code> folder and extract the <i>G3DCV2021_data</i> archive there.

# Libraries
Here the commands to import required libraries with [conda](https://conda.io/)  
<code>conda install numpy matplotlib</code>  
<code>conda install -c conda-forge opencv</code>  <small>Note: must be 4.2.0 or greater</small>  
<code>conda install -c anaconda scipy</code>  
<code>conda install ffmpeg-python</code>  
<code>conda install -c conda-forge pysoundfile</code>  
<code>conda install -c conda-forge moviepy</code>  
<code>conda install -c conda-forge svglib</code>  
<code>conda install -c anaconda reportlab</code>  
<code>conda install scikit-image</code>  

