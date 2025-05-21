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
docker run --shm-size=8g --gpus all -it --rm -v .:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
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
Download `ckpt` from the [link](https://drive.google.com/drive/folders/1WcOUPaSRRIENU-U1SQpC41WZz2nPP4iH?usp=sharing) and put it under `./CLIP_nodule`.
- In each folder, fold_X, it contains a checkpoint file `best_both.pt` for fold X, which will be loaded during evaluation.
- `args.text` contains arguments for training.

```bash
# You can also download it using gdown
pip install gdown
gdown --folder gdown --folder https://drive.google.com/drive/folders/1V1bUAt3Hl2WNh5eZmQCZHDqQmEd1FT7W?usp=sharing
```

### Data Requirement
To prepare a CSV file, list the path to the **NIfTI** file under the `image_path` column, along with the corresponding `pid`. 

The CSV file should contain six columns:  
| `pid` | `image_path` |  
|------|------------|  
| 001  | `./data/image1.nii.gz` |  
| 002  | `./data/image2.nii.gz` |  

Refer to `./dataset_csv/sample.csv` or `./dataset_csv/sample_paper.csv` as an example. `./dataset_csv/sample_paper.csv` contains three examples shown in Figure 2 in the manuscript.

Sample data can also be downloaded from the [link](https://drive.google.com/drive/folders/1elGnhviQBP8y7oPL2TpTn5jcBLE5HDs9?usp=drive_link).
```bash
# You can also download it using gdown
gdown --folder https://drive.google.com/drive/folders/1tQ_eD6i30C-qY9dfX4X20zuSyN7eB0lT?usp=drive_link
```
## Lung Segmentation
### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --csv_file ./dataset_csv/sample.csv --result_dir ./output --exp_dir ./model_weights --n 100 --thresh 0.55 --save_as nifti
```
#### Arguments
| Argument      | Type  | Default | Description |
|--------------|------|---------|-------------|
| `--csv_file`  | str  | `./dataset_csv/sample.csv` | Path to the CSV file containing image paths. |
| `--result_dir` | str  | `./output` | Directory to save the results. |
| `--exp_dir` | str  | `./model_weights` | Path to the experiment directory containing model weights. |
| `--n` | int | `100` | Number of masked samples to process. |
| `--thresh` | float | `0.55` | Threshold for segmentation. |
| `--save_as` | str | `nifti` | Output format (`nifti` or `numpy`). |

**Note:** The values `n=100` and `threshold=0.55` are used as the default values in the script. These parameters are also used to produce the results that are shown in the paper. You can adjust these values.

#### Outputs

By default, the segmentation results will be saved in the `./output` folder.  

- If saved as **NIfTI** files (`--save_as nifti`):  
  - A separate folder will be created for each `pid`.  
  - Inside each folder, the following files will be saved:  
    - `mean.nii.gz` – The average segmentation result across n masked samples.  
    - `std.nii.gz` – The standard deviation of the segmentation.  
  - Saving as **NIfTI** files may take longer.

- If saved as **NumPy** files (`--save_as numpy`):  
  - A single `.npz` file will be saved for each `pid` in the output directory.  
  - The file format will be `{pid}.npz`, containing:  
    - **First array** – The average segmentation result across n masked samples.  
    - **Second array** – The standard deviation.  
  - The segmentation results are saved in RAS orientation.

#### Visualization
Use `./tutorial/visualization.ipynb` to visualize the the predicted segmentation in jupyter notebook.



## Acknowledgements
This project is based on the code from the following repository:
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [dynamic_network_architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)

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
