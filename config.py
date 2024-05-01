import torch

debug = False
coord_path = '/workspace/radraid/projects/imaging_biomarker/CLIP_nodule/dataset_csv/img_coords.csv'
semantic_path = '/workspace/radraid/projects/imaging_biomarker/CLIP_nodule/dataset_csv/semantic_label.csv'
result_path = '/workspace/radraid/projects/imaging_biomarker/CLIP_nodule/results_loss'

batch_size = 4
num_workers = 4
lr = 1e-5
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 100
ga = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_embedding = 4096
max_length = 200

pretrained = True # for both image encoder and text encoder
freeze = True # for both image encoder and text encoder

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1

es_warmup = 0
es_patience = 10

general=True
internal=True
external=True
jitter = 10