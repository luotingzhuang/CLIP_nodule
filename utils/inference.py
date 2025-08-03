import os
import numpy as np
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Arg:
    def __init__(self):
        pass


def load_args(model_path: str):
    """
    Load the arguments from the model path.
    Args:
        model_path (str): Path to the model
    Returns:
        args: Arguments loaded from the saved model
    """

    with open(os.path.join(model_path, "args.txt"), "r") as file:
        # Read the entire file content
        args_text = file.read()
    args_text = args_text.replace("Namespace(", "").replace(")", "").split(", ")

    # Create a dictionary from the matches
    args_dict = {
        arg.split("=")[0]: arg.split("=")[-1].replace("'", "") for arg in args_text
    }

    args = Arg()
    args.dataset_path = args_dict["dataset_path"]
    args.img_dir = args_dict["img_dir"]
    args.split_dir = args_dict["split_dir"]
    args.text_dir = args_dict["text_dir"]
    args.n_splits = int(args_dict["n_splits"])

    args.jitter = 0
    args.model = args_dict["model"]
    args.tuning = args_dict["tuning"]
    args.dropout = 0.1
    args.class_weights = np.array([1, 1])
    args.clip_loss_weight = float(args_dict["clip_loss_weight"])
    args.img_loss_weight = float(args_dict["img_loss_weight"])
    args.text_loss_weight = float(args_dict["text_loss_weight"])
    args.weighted = args_dict["weighted"]
    args.debug = False
    args.tau = float(args_dict["tau"])
    args.out_dim = int(args_dict["out_dim"]) if "out_dim" in args_dict else 256

    if args.tuning == "lora":
        args.encoder = args_dict["encoder"]
        args.position = args_dict["position"]
        args.r = int(args_dict["r"])
        args.alpha = int(args_dict["alpha"])
        args.dropout_rate = float(args_dict["dropout_rate"])
        args.backbone = args.model.split("_")[1]
        if "biomedclip" in args.model:
            args.params = ["qkv", "query", "key", "value"]
        else:
            args.params = ["q", "k", "v"]

    print(args.__dict__)
    return args


def eval_epoch(model, valid_loader):
    """
    Evaluate the model on the test set.
    Args:
        model: CLIP model
        valid_loader: DataLoader for the test set
    Returns:
        labels: True labels
        logits_img_all: Logits for the images
        prob_img: Probabilities for the images
    """

    logits_img_all = []

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        img_embeds = model.encode_image(batch[0].to(device))
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        logits_img = model.classifier_image(img_embeds)
        logits_img_all.append(logits_img.cpu().detach())

    logits_img_all = torch.cat(logits_img_all, dim=0)
    prob_img = torch.softmax(logits_img_all, dim=-1)

    return prob_img.cpu().numpy()
