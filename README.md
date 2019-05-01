# DeepFusion-VO: Fusing Intensity and Event Frames for End-to-End Deep Visual Odometry

This is the project which [@epiception](https://github.com/epiception), [@SuhitK](https:://github.com/SuhitK) and I worked on for the Robot Localization and Mapping (16-833) course project at CMU in Spring 2019. The motivation of the project is to see how [DeepVO](http://senwang.gitlab.io/DeepVO/) can be enhanced using event-frames. To read more about event cameras and event SLAM, see this [link](http://rpg.ifi.uzh.ch/research_dvs.html).

## Installation Instructions

This is a `PyTorch` implementation. We assume `PyTorch` and dependencies are setup. This code has been tested on `PyTorch 0.4.1` with `CUDA 9.1` and `CUDNN 7.1.2`.

## Dependencies

We have a dependency on a few packages, especially `tqdm`, `scikit-image`, `tensorbordX` and `matplotlib`. They can be installed using standard pip as below:

```
pip install scipy scikit-image matplotlib tqdm natsort tensorboardX
```

To replicate the `conda` environment that we used for our training and evaluation

```
conda env create -f requirements.yml
```

## Datasets

This model assumes the [MVSEC](https://daniilidis-group.github.io/mvsec/) datasets available from the [Daniilidis Group](https://daniilidis-group.github.io/) at [University of Pennsylvania](https://www.upenn.edu/). The code to sync the dataframes for event and intensity frames along with poses will be released soon.

## Running Code

To run the code for base DeepVO results - without any fusion, from the base directory of the repository, run:

```
sh exec.sh
```

To run the fusion model, from the base directory of the repository, run:

```
sh exec_fusion.sh
```