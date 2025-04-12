# CSL7770_Assignment3
The repository contains the code and report for the third assignment of the course CSL7770 - Speech Understanding.

**Directory Structure**
```
.
├── paper.pdf
├── review.pdf
├── code
│   └── ..
└── README.md
```

`paper.pdf` is the original research paper.

`review.pdf` contains a summary, and review of the paper as part of the assignment.

The code directory contains all the code related to the implementation.

**How to use this repository**

First, clone the repository to your local machine and navigate to the repository using the following commands:
```bash
git clone https://github.com/Shubh-Goyal-07/CSL7770_Assignment_3.git
cd CSL7770_Assignment_3
```


Navigate to the `code` directory to run the code.

1. **Navigate to the code directory**
   ```bash
   cd code
   ```
    This directory contains all the code files.

2. **Prepare the environment**
   ```bash
   conda create -n csl7770 -y
   conda activate csl7770
   conda install --file requirements.txt -y -c conda-forge -c nvidia -c pytorch
   ```


## Code Details

- The current implementation does **not** include any dataset used in the original paper.
- Instead, the model has been tested on a **randomly generated dummy dataset**.
- The model can be trained using:
  
  ```bash
  python train.py  --epochs <number_of_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --dataset <dataset_name>
  ```

  The dataset name can be `dummy` or `your_dataset_name`. The `--dataset` argument is used to specify the dataset to be used for training. The `--epochs`, `--batch_size`, and `--learning_rate` arguments are used to specify the number of epochs, batch size, and learning rate for training, respectively.

  A model with the name `lstm_filter_<dataset_name>.pth` will be saved in the `models` directory after training.

- To use another dataset, we need to add a dataset class in `datasets.py` and ensure the output format matches that of the dummy dataset.

## Inference

- Run inference using:

  ```bash
  python inference.py
  ```

- The inference does not yet contain the code for **file loading for actual `.mp3` or `.wav` input**  The current version has been tested on a **randomly generated sample input**.