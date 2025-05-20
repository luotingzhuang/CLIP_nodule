import torch
from models.CLIP import CLIPModel
from loralib.utils import apply_lora, mark_only_lora_as_trainable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(args):
    '''
    Initialize the model based on the provided arguments.
    Args:
        args: Arguments loaded from the saved model
    Returns:
        model: CLIP model
    '''
    model = CLIPModel(args)

    if 'ft' in args.tuning:
        print("Full tuning")
    elif args.tuning == 'pt':
        print("Probe tuning")
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'image_projection' in name or 'semantic_projection' in name or 'classifier_image' in name or 'classifier_text' in name or 'logit_scale' in name:
                param.requires_grad = True

    elif args.tuning == 'lora':
        print("LoRA tuning")
        args.backbone = args.model.split('_')[1]
        list_lora_layers = apply_lora(args, model)
        mark_only_lora_as_trainable(model)

        for name, param in model.named_parameters():
            if 'image_projection' in name or 'semantic_projection' in name or 'classifier_image' in name or 'classifier_text' in name or 'logit_scale' in name:
                param.requires_grad = True

    model = model.to(device)
    return model