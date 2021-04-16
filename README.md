# Multi label ingredient classification:

Initially cloned from: https://github.com/dfan/awa2-zero-shot-learning

## Data

Vireo-172 Dataset: http://vireo.cs.cityu.edu.hk/VireoFood172

## Model

Resnet-50 backbone with a linear classifier. 

## Results

Micro-F1: 0.72
Macro-F1: 0.57

## Usage

### Training

Unzip the data from Vireo172 into a new directory titled data and run:
python train.py

### Testing

Make sure you have a model.bin file in checkpoint and run 
python train.py --mode test
