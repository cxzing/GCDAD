# GCDAD
## Requirements
The recommended requirements for GCDAD are specified as follows:
- `arch==6.1.0`
- `einops==0.6.1`
- `matplotlib==3.7.0`
- `numpy==1.23.5`
- `pandas==1.5.3`
- `Pillow==9.4.0`
- `scikit_learn==1.2.2`
- `scipy==1.8.1`
- `statsmodels==0.14.0`
- `torch==1.13.0`
- `tqdm==4.65.0`
- `tsfresh==0.20.1`

The dependencies can be installed by running the following command:

```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and placed into the `datasets/` folder as follows:

- Our model supports anomaly detection for both univariate and multivariate time series datasets.
- We provide the MSL and PSM dataset.You can obtain the dataset from the following link:<https://drive.google.com/drive/folders/1DDee8zg0dwfXWqkruppOW_AnjQrjfy5g?usp=drive_link> If you want to use your own dataset, please place your dataset files in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, and `<dataset>_test_label.npy`.

## Code Description

There are six files/folders in the source:

- **data_factory**: The preprocessing folder/file. All datasets preprocessing codes are here.
  
- **main.py**: The main python file. You can adjust all parameters in there.
  
- **metrics**: There is the evaluation metrics code folder.
  
- **model**: GCDAD model folder.
  
- **solver.py**: The training and testing processing are all in there.
  
- **requirements.txt**: Python packages needed to run this repo.


## Usage

1.Install
- Python 3.9
- PyTorch >= 1.4.0

2.Download the datasets

3.Train and Evaluate GCDAD on a Dataset
To train and evaluate GCDAD on a dataset, run the following command:

```bash
python main.py
```


