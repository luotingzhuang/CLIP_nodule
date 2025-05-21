# Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Early Lung Cancer Detection

[![arXiv](https://img.shields.io/badge/arXiv-2504.21344-b31b1b.svg)](https://arxiv.org/abs/2504.21344)



**Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ramin Salehi-Rad, Linh M. Tran, Denise R. Aberle, Ashley E. Prosper, William Hsu**

<sup>1</sup>Medical & Imaging Informatics, Department of Radiological Sciences, David Geffen School of Medicine at UCLA, Los Angeles, CA;  

<sup>2</sup>Department of Medicine, Division of Pulmonology and Critical Care, David Geffen School of Medicine at UCLA, Los Angeles, CA


<p align="center">
    <img src="figures/abstract.jpg" width="80%"/> <br />
    <em> 
    Figure 1. An overview of the proposed framework. 
    </em>
</p>

**Objective:** A number of machine learning models have utilized semantic features, deep features, or both to assess lung nodule malignancy. However, their reliance on manual annotation during inference, limited interpretability, and sensitivity to imaging variations hinder their application in real-world clinical settings. Thus, this research aims to integrate semantic features derived from radiologists’ assessments of nodules, allowing the model to learn clinically relevant, robust, and explainable features for predicting lung cancer. 

**Methods:** We obtained 938 low-dose CT scans from the National Lung Screening Trial with 1,246 nodules and semantic features. Additionally, the Lung Image Database Consortium dataset contains 1,018 CT scans, with 2,625 lesions annotated for nodule characteristics. Three external datasets were obtained from UCLA Health, the LUNGx Challenge, and the Duke Lung Cancer Screening. For imaging input, we obtained 2D nodule slices from nine directions from 50×50×50mm nodule crop. We converted structured semantic features into sentences using Gemini. We finetuned a pretrained Contrastive Language-Image Pretraining model with a parameter-efficient fine-tuning approach to align imaging and semantic features and predict the one-year lung cancer diagnosis.  

**Results:** We evaluated the performance of the one-year diagnosis of lung cancer with AUROC and AUPRC and compared it to three state-of-the-art models. Our model demonstrated an AUROC of 0.90 and AUPRC of 0.78, outperforming baseline state-of-the-art models on external datasets. Using CLIP, we also obtained predictions on semantic features, such as nodule margin (AUROC: 0.81), nodule consistency (0.81), and pleural attachment (0.84), that can be used to explain model predictions.  

**Conclusion:** Our approach accurately classifies lung nodules as benign or malignant, providing explainable outputs, aiding clinicians in comprehending the underlying meaning of model predictions. This approach also prevents the model from learning shortcuts and generalizes across clinical settings. 

## Getting Started
### Create a Docker Container
```bash
docker run --shm-size=8g --gpus all -it --rm -v .:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:24.03-py3
```
- If you use `-v .:/workspace` as shown above, Docker will map the **current directory** to `/workspace` inside the container.
- To map a different folder to a specific path in docker container, you can replace `-v .:/workspace` with `-v /path/to/local/folder:/path/in/container`.

### Clone the Repository and Install Packages
1. Go to the folder you want to store the code and clone the repo
```bash
git clone https://github.com/luotingzhuang/CLIP_nodule.git
cd CLIP_nodule
```

2. Install all of the required python packages using the following command line.
```bash
pip install -r requirements.txt
```

### Download Pretrained Weights
Download `ckpt` from the [link](https://drive.google.com/drive/folders/1V1bUAt3Hl2WNh5eZmQCZHDqQmEd1FT7W?usp=sharing) and put it under `./CLIP_nodule`.
- In each folder, fold_X, it contains a checkpoint file `best_both.pt` for fold X, which will be loaded during evaluation.
- `args.text` contains arguments for training.

```bash
# You can also download it using gdown
pip install gdown
gdown --folder --fuzzy --no-cookies --no-check-certificate https://drive.google.com/drive/folders/1V1bUAt3Hl2WNh5eZmQCZHDqQmEd1FT7W?usp=sharing
```

### Data Requirement
To prepare a CSV file, list the path to the **NIfTI** file under the `image_path` column, along with the corresponding `pid` and `nodule_id`. `coordX`, `coordY`, and `coordZ` are the nodule centroid in a global coordinate system. These can be extracted from the nodule mask using the [code](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/master/tutorials/get_seed_from_mask.ipynb). If the nodule mask is not available, we recommend using a nodule detection algorithm, such as [monai nodule detection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection) to obtain the nodule location from CT scans.

The CSV file should contain six columns:  
| pid    | nodule_id | image_path                                      | coordX     | coordY     | coordZ      | 
|--------|-----------|--------------------------------------------------|------------|------------|-------------|
| 121389 | 0         | ./sample_data/121389/2001-01-02/image.nii.gz     | -38.038567 | -73.942905 | -111.030769 |

Refer to `./dataset_csv/sample_csv.csv` as an example. Note that the `malignant` column is optional.

Sample data can also be downloaded from the [link](https://drive.google.com/drive/folders/1MhcOCLpG1OrdGyQw9OiwNQELZKfIBGlr?usp=drive_link).
```bash
# You can also download it using gdown
gdown --folder --fuzzy --no-cookies --no-check-certificate https://drive.google.com/drive/folders/1MhcOCLpG1OrdGyQw9OiwNQELZKfIBGlr?usp=drive_link
```

## Predict Nodule Malignancy
### Preprocessing
Before running inference, we need to crop a 100×100×100 mm bounding box around the nodule and save the resulting cropped volume as a `.pt` file for later use.

```bash
python crop_nodule.py --dataset_path ./dataset_csv/sample_csv.csv --save_path ./cropped_img
```
| Argument      | Type  | Default | Description |
|--------------|------|---------|-------------|
| `--dataset_path`  | str  | `./dataset_csv/sample_csv.csv` | Path to the CSV file containing image paths and nodule centroid. |
| `--num_workers` | int  | 4 | Number of workers for data loading. |
| `--exp_dir` | str  | `./cropped_img` | Path to save the cropped nodules. |

The nodule crop will be saved with the format `{pid}_{nodule_id}.pt`.

### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path ./ckpt --dataset_path ./dataset_csv/sample_csv.csv --img_dir ./cropped_img --num_workers 4 --save_path ./results_csv
```
#### Arguments
| Argument      | Type  | Default | Description |
|--------------|------|---------|-------------|
| `--model_path`  | str  | `./ckpt` | Path to the model checkpoint folder. |
| `--dataset_path` | str  | `./dataset_csv/sample_csv.csv` | Path to the CSV file pid and nodule id. |
| `--img_dir` | str  | `./cropped_img` | Path to save the cropped nodules from the preprocessing step. |
| `--num_workers` | int | `4` | Number of workers for data loading. |
| `--save_path` | str | `./results_csv` | Path to save the results. |

The output CSV file will be saved at `{save_path}/result.csv`. It includes the pid and the corresponding predicted probabilities. Each `raw_X` column represents the probability output from fold X, while the `ensemble` column contains the average probability across all folds.

## Acknowledgements
This project is based on the code from the following repository:
- [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA.git)
- [foundation-cancer-image-biomarker](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/tree/master)

## TODO
- [ ] Training code

## CITATION
```bibtex
@article{zhuang2025vision,
  title={Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Early Lung Cancer Detection},
  author={Zhuang, Luoting and Tabatabaei, Seyed Mohammad Hossein and Salehi-Rad, Ramin and Tran, Linh M and Aberle, Denise R and Prosper, Ashley E and Hsu, William},
  journal={arXiv preprint arXiv:2504.21344},
  year={2025}
}
```

## CONTACT
If you have any questions, please don't hesitate to contact us at luotingzhuang@g.ucla.edu.
