# SCFMUNet 

## 1.Running Environments

python=3.8

torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cuda >==11.7

timm==0.4.12

triton==2.0.0

causal_conv1d==1.0.0  

mamba_ssm==1.0.1   

## 2.datasets

### ACDC

For the ACDC dataset, you could download them from [Baidu](https://pan.baidu.com/s/1WQeEfAiUZCnrPGxXmbuJSg?pwd=gtv3). After downloading the datasets, you are supposed to put them into './data/ACDC/', and the file format reference is as follows.

- `./data/Synapse/`
  - `lists_ACDC`
    - `test.txt`
    - `train.txt`
    - `valid.txt`
  - `test`
    - `case_xxx_volume_ED.npz`
    - `case_xxx_volume_ES.npz`
  - `train`
    - `case_xxxsliceED_x.npz`
    - `case_xxxsliceES_x.npz`
  - `valid`
    - `case_xxxsliceED_x.npz`
    - `case_xxxsliceES_x.npz`

### Synapse

For the Synapse dataset, you could download them from [Baidu](https://pan.baidu.com/s/1dybSiRd1uT2h4X3Td9vn8g?pwd=63rw). After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- After downloading the datasets, you are supposed to put them into `./data/Synapse/`, and the file format reference is as follows.

- `./data/Synapse/`
  - `lists`
    - `list_Synapse`
      - `all.lst`
      - `test_vol.txt`
      - `train.txt`
  - `test_vol_h5`
    - `casexxxx.npy.h5`
  - `train_npz`
    - `casexxxx_slicexxx.npz`
    
## 3.Prepare the pre_trained weights

You could download them from [Baidu](https://pan.baidu.com/s/1L1tcgwasrVY89MwLPaUOSA?pwd=3ixa) and put them into './pre_trained_weights/'.

## 4.How to Run

python ACDC_train.py or python train.py 
