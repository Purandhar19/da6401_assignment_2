# DA6401 Assignment 2 - Visual Perception Pipeline

Implementation of a multi-stage visual perception pipeline on the Oxford-IIIT Pet Dataset using PyTorch.

## Links

- **W&B Report**: https://wandb.ai/ae22b069-iit-madras/da6401-assignment2/reports/da6401_assignment_2_report--VmlldzoxNjQ4OTc0Ng
- **GitHub Repo**: https://github.com/Purandhar19/da6401_assignment_2.git

## Tasks Implemented

- **Task 1**: VGG11 classification from scratch with BatchNorm and CustomDropout (37 pet breeds)
- **Task 2**: Bounding box localization using VGG11 encoder + regression head with custom IoU loss
- **Task 3**: U-Net style semantic segmentation with transposed convolution decoder and skip connections
- **Task 4**: Unified multi-task pipeline with shared backbone and single forward pass

## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   ├── __init__.py
│   └── iou_loss.pyc
├── models/
│   ├── __init__.py
│   ├── classification.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── inference.py
├── multitask.py
├── README.md
├── requirements.txt
└── train.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Run all experiments
python train.py --data_root /path/to/oxford-iiit-pet --run all

# Run specific task
python train.py --data_root /path/to/oxford-iiit-pet --run task1
python train.py --data_root /path/to/oxford-iiit-pet --run task2
python train.py --data_root /path/to/oxford-iiit-pet --run task3
```

## Dataset

Oxford-IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
