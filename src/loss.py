import torch
import torch.nn.functional as F


def infonce_hybrid_loss(alpha, s_d_pos, s_s_pos, s_d_negs, s_s_negs, tau=0.05):
    """
    Calculate the InfoNCE hybrid loss combining positive and negative similarity scores.

    This function implements a variant of the InfoNCE loss where the positive and
    negative similarity scores are interpolated based on a weighting factor `alpha`.
    The positive similarity score is adjusted by combining `s_d_pos` (document score)
    and `s_s_pos` (semantic score). Similarly, the negative similarity scores are
    modified by interpolating `s_d_negs` (document scores) and `s_s_negs` (semantic
    scores). The modified scores are used to compute the cross-entropy loss for
    contrastive learning.

    :param alpha: Weighting factor for interpolation between document-based
                  and semantic-based similarity scores. It should be a float
                  tensor.
    :param s_d_pos: Positive similarity score calculated using document-based
                    features, with shape (batch_size,).
    :param s_s_pos: Positive similarity score calculated using semantic-based
                    features, with shape (batch_size,).
    :param s_d_negs: Negative similarity scores calculated using document-based
                     features, with shape (batch_size, num_negatives).
    :param s_s_negs: Negative similarity scores calculated using semantic-based
                     features, with shape (batch_size, num_negatives).
    :param tau: Temperature scaling factor for logits. The default value is 0.1.

    :return: The computed InfoNCE hybrid loss as a single scalar tensor.
    :rtype: torch.Tensor
    """

    hybrid_pos = (alpha * s_d_pos + (1.0 - alpha) * s_s_pos).unsqueeze(1)
    alpha_expanded = alpha.unsqueeze(1)
    hybrid_negs = alpha_expanded * s_d_negs + (1.0 - alpha_expanded) * s_s_negs
    logits = torch.cat([hybrid_pos, hybrid_negs], dim=1) / tau
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)

    return loss
