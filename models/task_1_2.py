import torch
import torch.nn as nn
from modules import TransformerEncoder, ClassificationHead, MCCFormers_D, RetfoundVit


class Task1Model(nn.Module):

    def __init__(self, feature_dim=1024, dropout=0.4, h=14, w=14, d_model=512, n_head=8, n_layers=2, num_classes=4, use_pe_3=False, local_attn_size=None):
        super(Task1Model, self).__init__()

        self.feature_extractor = RetfoundVit()

        self.change_encoder = MCCFormers_D(
            feature_dim=feature_dim,
            dropout=dropout,
            h=h, w=w, d_model=d_model, n_head=n_head, n_layers=n_layers, local_attn_size=local_attn_size
        )

        self.classification_encoder = TransformerEncoder(
            feature_dim=feature_dim,
            dropout=dropout,
            h=h,
            w=w,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            use_pe=use_pe_3
        )

        self.cls_head = ClassificationHead(in_channels=d_model, num_classes=num_classes)

        self.feature_dim = feature_dim
        self.w = w
        self.h = h

    def forward(self, img_t0, img_t1):
        batch = img_t0.size(0)

        features_t0 = self.feature_extractor(img_t0)  # (batch, w*h, feature_dim)
        features_t1 = self.feature_extractor(img_t1)  # (batch, w*h, feature_dim)

        features_t0 = features_t0.permute(0, 2, 1)
        features_t0 = features_t0.view(batch, self.feature_dim, self.h, self.w)  # (batch, feature_dim, h, w)

        features_t1 = features_t1.permute(0, 2, 1)
        features_t1 = features_t1.view(batch, self.feature_dim, self.h, self.w)  # (batch, feature_dim, h, w)

        features = self.change_encoder(features_t0, features_t1)  #  (batch_size, 2*d_model, h, w)
        features = self.classification_encoder(features)  #  (batch_size, d_model)

        logits = self.cls_head(features)

        return logits


class Task2Model(nn.Module):

    def __init__(self, feature_dim=1024, dropout=0.4, h=14, w=14, d_model=512, n_head=8, n_layers=2, num_classes=3, use_pe_3=False):
        super(Task2Model, self).__init__()

        self.feature_extractor = RetfoundVit()

        self.classification_encoder = TransformerEncoder(
            feature_dim=feature_dim,
            dropout=dropout,
            h=h,
            w=w,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            use_pe=use_pe_3
        )

        self.cls_head = ClassificationHead(in_channels=d_model, num_classes=num_classes)

        self.feature_dim = feature_dim
        self.w = w
        self.h = h

    def forward(self, img):
        batch = img.size(0)

        features = self.feature_extractor(img)  # (batch, w*h, feature_dim)

        features = features.permute(0, 2, 1)
        features = features.view(batch, self.feature_dim, self.h, self.w)  # (batch, feature_dim, h, w)

        features = self.classification_encoder(features)  #  (batch_size, d_model)

        logits = self.cls_head(features)

        return logits

