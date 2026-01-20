import torch
from torch import nn
from torch.nn import functional as F
from args import args

class SupConLoss(nn.Module):  # Pretraining loss (multi-view SupCon)
    """Multi-view SupCon: accepts features as [B, V, D]"""
    def __init__(self, temperature=0.06, device="cuda:0", usetcr=True):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.tcr_fn = TotalCodingRate(eps=args.eps)
        self.use_tcr = usetcr

    def forward(self, features, labels=None):
        """
        features: [B, V, D] embeddings for V views per sample
        labels:   [B] class labels for each ORIGINAL sample
        """

        # Normalize embeddings (cosine similarity via dot product)
        features = F.normalize(features, dim=-1)

        B, V, D = features.shape

        # Build sample-level positive mask [B,B]
        # mask[i,j]=1 if same label (Supervised), else 0
        labels = labels.contiguous().view(-1, 1)  # [B,1]
        mask = torch.eq(labels, labels.T).float().to(self.device)  # [B,B]

        # Flatten views -> [VB, D]
        # Index ordering: all view0 (B rows), then view1 (B rows), ...
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [V*B, D]

        # Similarity logits for all pairs: [VB, VB]
        logits = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Expand mask from [B,B] to [VB,VB]
        mask = mask.repeat(V, V)  # [V*B, V*B]

        # Remove self-comparisons
        VB = V * B
        logits_mask = torch.ones((VB, VB), device=self.device, dtype=torch.float32)
        logits_mask.fill_diagonal_(0.0)

        # positives are same-label pairs excluding self
        mask = mask * logits_mask

        # Log-softmax over all n != i
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Average log-prob over positives for each anchor
        denom = mask.sum(1)
        denom = torch.clamp(denom, min=1.0)  # safety (avoid div-by-zero)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom

        # Final loss averaged over all anchors
        loss = -mean_log_prob_pos
        loss = loss.view(V, B).mean()

        # Optional TCR regularizer: average TCR over views
        if self.use_tcr:
            tcr = 0.0
            for v in range(V):
                tcr = tcr + self.tcr_fn(features[:, v, :])
            tcr = tcr / V
            loss = loss + args.alpha * tcr

        return loss


class TotalCodingRate(nn.Module):  # Regularization for pretraining loss
    """credits to: https://github.com/tsb0601/EMP-SSL"""
    def __init__(self, eps=1.):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):
        return -self.compute_discrimn_loss(X.T)


class ProtoCLR(nn.Module): #Finetuning loss
    def __init__(self, tau=1.):
        super(ProtoCLR, self).__init__()
        self.tau = tau

    def forward(self, z1_features, z2_features, labels):

        labels = torch.cat([labels, labels], dim=0)

        z1_features, z2_features = F.normalize(z1_features), F.normalize(z2_features)

        z_features = torch.cat([z1_features, z2_features])

        unique_labels = torch.unique(labels)

        # Compute similarity between each feature and its corresponding class mean
        feature_means = torch.stack([z_features[labels == label].mean(dim=0) for label in unique_labels])

        feature_means_repeated = torch.zeros((z_features.shape[0], z_features.shape[1])).to(z_features.device)
        for label in torch.unique(labels):
            feature_means_repeated[labels==label] = torch.mean(z_features[labels==label], dim=0)

        sim_proto = torch.diag(torch.mm(feature_means_repeated, z_features.T)) / self.tau

        sim_all = torch.mm(z_features, feature_means.T) / self.tau

        # Formulate the loss as NT-Xent
        exp_sim = torch.exp(sim_all - sim_proto.unsqueeze(1).repeat(1, sim_all.shape[1])) # removing the pos sim from denominator
        log_prob_pos = sim_proto - torch.log(exp_sim.sum(1))
        loss = - log_prob_pos.mean()

        return loss