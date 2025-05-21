import torch
from torch import nn
import torch.nn.functional as F
import clip
from models.modules import ProjectionHead, Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.model = args.model
        self.class_weights = args.class_weights
        self.tuning = args.tuning
        self.clip_loss_weight = args.clip_loss_weight
        self.img_loss_weight = args.img_loss_weight
        self.text_loss_weight = args.text_loss_weight

        visual_encoder_name = args.model.split("_")[1]
        print("Loading openai clip...")
        model, _ = clip.load(visual_encoder_name, device=device, jit=False)
        self.positional_embedding = model.positional_embedding

        model = model.float()
        self.vision_model = model.visual
        self.token_embedding = model.token_embedding

        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.dtype = model.dtype
        self.text_projection = model.text_projection

        self.image_projection = Attention(512, M=256, L=128)
        self.semantic_projection = ProjectionHead(
            embedding_dim=512, projection_dim=256, dropout=args.dropout
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / args.tau)))

        self.classifier_image = nn.Linear(256, 2)
        self.classifier_text = nn.Linear(256, 2)

        self.pred_criterion_train = nn.CrossEntropyLoss()
        self.pred_criterion_val = nn.CrossEntropyLoss()

    def encode_text(self, input_ids=None):
        """
        Encodes the text input using the CLIP model.
        Args:
            input_ids: The input text ids.
        Returns:
            text_embeds: The encoded text embeddings.
        """
        x = self.token_embedding(input_ids).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection
        text_embeds = self.semantic_projection(x)

        return text_embeds

    def encode_image(self, pixel_values=None):
        """
        Encodes the 2D slices using the CLIP model.
        Args:
            pixel_values: The input image pixel values.
        Returns:
            img_embeds: The encoded image embeddings.
        """

        img_embeds = []
        for pixel_val in pixel_values:
            vision_output_i = self.vision_model(pixel_val)
            img_embeds_i = self.image_projection(vision_output_i.float())
            img_embeds.append(img_embeds_i)
        img_embeds = torch.cat(img_embeds, 0)
        return img_embeds

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()
        labels = kwargs["labels"].type(torch.LongTensor).cuda()
        mode = kwargs["mode"]
        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # clip
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        loss_clip = self.clip_loss(logits_per_text)

        clip_acc_text = self.compute_accuracy(logits_per_image)
        clip_acc_images = self.compute_accuracy(logits_per_text)

        # malignancy prediction
        logits_img = self.classifier_image(img_embeds)
        logits_text = self.classifier_text(text_embeds)

        n_classifiers = 2.0

        # remove na
        logits_img = logits_img[labels >= 0, :]
        logits_text = logits_text[labels >= 0, :]
        labels = labels[labels >= 0]
        if len(labels) == 0:
            loss_pred = torch.tensor(0.0).cuda()
        else:
            if mode == "train":
                loss_pred = (
                    self.pred_criterion_train(logits_img, labels) * self.img_loss_weight
                )
                loss_pred += (
                    self.pred_criterion_train(logits_text, labels)
                    * self.text_loss_weight
                )

            else:
                loss_pred = (
                    self.pred_criterion_val(logits_img, labels) * self.img_loss_weight
                )
                loss_pred += (
                    self.pred_criterion_val(logits_text, labels) * self.text_loss_weight
                )

        loss_pred = loss_pred / n_classifiers

        return (
            loss_clip,
            loss_pred,
            clip_acc_text,
            clip_acc_images,
            logits_img,
            logits_text,
        )

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:

        targets = torch.arange(similarity.size(0)).cuda()
        texts_loss = self.clip_criterion(similarity, targets, smoothing=0.1)
        images_loss = self.clip_criterion(similarity.t(), targets, smoothing=0.1)
        return (texts_loss + images_loss) / 2.0 * self.clip_loss_weight

    def compute_accuracy(self, logits):
        gt = torch.arange(logits.size(0)).cuda()
        logits = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == gt).sum().item()
        total = gt.size(0)
        accuracy = correct / total
        return accuracy


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
