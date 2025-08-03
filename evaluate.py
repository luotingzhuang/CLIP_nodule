import torch
import pandas as pd
import os
import monai
import argparse
from betacal import BetaCalibration

from utils.utils import init_model
from dataset.dataset_visiononly import VisionDatasetText
from utils.inference import load_args, eval_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_args():
    """
    Load the arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--model_path", type=str, default="./ckpt", help="Path to the model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset_csv/sample_csv.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="./cropped_img",
        help="Path to the cropped images",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results_csv",
        help="Path to save the results",
    )

    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="best_both.pt",
        help="Name of the checkpoint file to load",
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Whether to use calibration for evaluation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    eval_args = load_eval_args()
    args = load_args(eval_args.model_path)
    args.dataset_path = eval_args.dataset_path
    args.img_dir = eval_args.img_dir
    exp_name = eval_args.model_path.split("/")[-1]

    # DataLoaders
    test_dataset = VisionDatasetText(args)
    test_loader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=eval_args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    model = init_model(args)
    all_test_result = []

    # Loop through folds
    for fold in range(args.n_splits):
        print(f"Loading fold {fold}...")
        try:
            weight_path = os.path.join(
                eval_args.model_path, f"fold_{fold}/{eval_args.ckpt_file}"
            )
            print("Loading model from", weight_path)
            pretrained_dict = torch.load(weight_path, map_location=device)["model"]
        except:
            print("Model path does not exist. Exiting...")
            exit()

        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        with torch.no_grad():
            probs = eval_epoch(model, test_loader)
            result = test_dataset.data_subset[["pid", "nodule_id"]].copy()
            result.loc[:, "probs"] = probs[:, 1]
            result.loc[:, "fold"] = fold
            all_test_result.append(result)

    # Organize results from all folds
    outputdf = pd.DataFrame(
        {f"raw_{i}": all_test_result[i].probs.values for i in range(5)}
    )
    outputdf.loc[:, "ensemble"] = outputdf.mean(1).values
    outputdf = pd.concat([all_test_result[0][["pid", "nodule_id"]], outputdf], axis=1)


    if eval_args.calibrate:
        print("Calibrating results...")
        import joblib
        for fold in range(args.n_splits):
            calibrator = joblib.load(
                os.path.join(eval_args.model_path, f"fold_{fold}/cal_{fold}.pkl")
            )
            outputdf[f"calibrated_{fold}"] = calibrator.predict(
                outputdf[f"raw_{fold}"]
            )
        outputdf["calibrated_ensemble"] = outputdf[
            [f"calibrated_{i}" for i in range(args.n_splits)]
        ].mean(axis=1)
        

    os.makedirs(eval_args.save_path, exist_ok=True)
    outputdf.to_csv(os.path.join(eval_args.save_path, 
                                 f"{exp_name}_{eval_args.ckpt_file.split('.')[0]}_result.csv"), 
                                 index=False)
    print(
        "Results saved to",
        os.path.join(
            eval_args.save_path,
            f"{exp_name}_{eval_args.ckpt_file.split('.')[0]}_result.csv",
        ),
    )
