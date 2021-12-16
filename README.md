# Smartphone-based-RTI
### Geometric and 3D Computer Vision 2021

See <b>[G3DCV2021_FinalProject.pdf](G3DCV2021_FinalProject.pdf)</b> for more details.  

## 1. Introduction
<b>Reflectance Transformation Imaging (RTI)</b> is a technique to capture the reflectance of an
object under different lighting conditions. It enables the interactive relighting of the subject
from any direction and the mathematical enhancement of subject’s surface shape and color
attributes. Typically, RTI uses a static camera and an array of light sources at known (or
knowable) positions.  

## 2. Assignment
In this project, you’ll explore RTI using a cheaper setup composed by two consumer
smartphones equipped with a camera and LED flashlight. The idea is to use one of the
two smartphones as a movable light source while the other captures the object from a
frontal position. We exploit the camera of the movable light source to recover the light
direction vector at each point and provide the same data obtainable with a light dome.

### 2.1. Requirements
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

### 2.2. Assets
Download the assets folder from [here](http://www.dsi.unive.it/~bergamasco/teachingfiles/G3DCV2021_data.7z).  
Create the <code>assets/</code> folder and extract the <b>G3DCV2021_data</b> archive on the root folder.

## 3. Setup Project
Here a small description of the developed project done, and how it works.  
First download the <code>assets</code> as described before.

### 3.1 Libraries
This project use <b>python 3.7</b>.  
Download the required project libraries.  
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
<code>conda install -c conda-forge python-dotenv</code>  

#### GPU Libraries (optional)
<small>Note: this is required to run on GPU</small>  
<code>conda install numba</code>  
<code>conda install cudatoolkit</code>  

### 3.2 Project Structure
The project contains the 3 different programs required from the assignment.  

#### Main programs
1. <code>camera_calibrator.py</code>: basic program that read the two chessboard images and save camera intrinsics into assets folder.
2. <code>analysis.py</code>: core program that perform different tasks, it can require some time to finish (1/2 hours):
   1. <b>sync_videos():</b> extract the audio offset between the static and moving video (required for sync)
   2. <b>generate_video_default_frame():</b> extract and save a default frame from static video (a frame with no light)
   3. <b>extract_video_frames():</b> extract the video frames with the class <b>FeatureMatcher</b>
   This class will compute the corner detection, intensity extraction and get the camera pose of given frames (static and moving).  
   Results are saved on storage and can be reused.
   4. <b>compute_intensities():</b> compute light vectors intensities for each frame pixel.
   5. <b>interpolate_intensities():</b> compute the interpolation of pixel intensities, two interpolation function are supported:
      1. Polynomial Texture Maps (<b>PTM</b>)
      2. Linear Radial Basis Function (<b>RBF</b>)
   6. <b>prepare_images_data():</b> prepare images for each camera position (normalized) with interpolated values. This process is required in order to speed up relighting program.
   7. Save interpolation results on storage (can require some GB)
3. <code>interactive_relighting.py</code>: the final program, require the analysis results in order to execute.
It reads the interpolation results and render the relighted image dynamically based on the light direction specified with the cursor.

#### Utils
1. <code>audio_utils.py</code>: contains utility functions regarding audio management.
2. <code>email_utils.py</code>: contains utility functions regarding email management.
3. <code>image_utils.py</code>: contains utility functions regarding image and video management.
4. <code>utilities.py</code>: contains misc utility functions.

### 3.3. Parameters
Each program has some tunable parameters on <b>compute()</b> function:
1. <code>camera_calibrator.py</code>: no tunable parameters.
2. <code>analysis.py</code>:
   1. <b>video_name</b>: specify the name of the video to analyze.
   2. <b>from_storage</b>: specify whether to read a previous <b>extract_video_frames()</b> results, in this case analysis will start from <b>compute_intensities()</b> phase.
   3. <b>storage_filepath</b>: specify a different path whether to read <b>extract_video_frames()</b> results
   4. <b>interpolate_PTM</b>: specify the interpolation function to use
   5. <b>notification_email</b>: specify if send a notification email when the computation is finish.  
   <b>Note</b> in order to use this parameter you should rename <code>.env.example</code> to <code>.env</code> and fill it with valid google account credentials. The env file is used in order to keep credentials off-repo.
   6. <b>debug</b>: specify whether to compute analysis on debug mode (analyze first pixel only on interpolation)
3. <code>interactive_relighting.py</code>:
   1. <b>video_name</b>: specify the name of the video to relight.
   <b>storage_filepath</b>: specify a different path whether to read <b>analysis()</b> results.
   
### 3.4. Run and Test
Each program have a <code>compute()</code> function.  
In order to test each program individually there is also a <code>main()</code> function, that will simply call the <b>compute()</b> function with some pre-set parameters.  
- To run all the 3 programs sequentially from skratch simply use <code>$ python main.py</code>.  
- To run them separately use <code>$ python {program_name}.py</code>, where <i>program_name</i> is one of the 3 main programs.