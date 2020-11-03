import torch


def _angular_difference(predicted_T1, predicted_T2, labels_T1, labels_T2):
    """Computes the angles between T1 and T2 and then angular differences between predictions and labels

    Args:
        predicted_T1 (torch.tensor): first half of predictions
        predicted_T2 (torch.tensor): second half of predictions
        labels_T1 (torch.tensor): first half of labels
        labels_T2 (torch.tensor): second half of labels

    Returns:
        [torch.tensor]: elementwise angular differences
    """
    # Compute acceleration norms
    norm_T1 = torch.norm(predicted_T1, dim=1)
    norm_T2 = torch.norm(predicted_T2, dim=1)
    norm_LT1 = torch.norm(labels_T1, dim=1)
    norm_LT2 = torch.norm(labels_T2, dim=1)

    # Compute normalized accelerations
    predicted_T1 = (predicted_T1.transpose(0, 1) / norm_T1).transpose(0, 1)
    predicted_T2 = (predicted_T2.transpose(0, 1) / norm_T2).transpose(0, 1)
    labels_T1 = (labels_T1.transpose(0, 1) / norm_LT1).transpose(0, 1)
    labels_T2 = (labels_T2.transpose(0, 1) / norm_LT2).transpose(0, 1)

    # Vectoried dot products
    pred_dots = torch.einsum('bs,bs->b', predicted_T1, predicted_T2)
    labels_dots = torch.einsum('bs,bs->b', labels_T1, labels_T2)

    # Compute angles between pairs
    pred_angles = torch.acos(pred_dots)
    label_angles = torch.acos(labels_dots)

    # Compute error in the angles
    return torch.abs(pred_angles-label_angles)


def contrastive_loss(predicted, labels):
    """ Will compute a normalized_loss and additionally a term that compares the error on the angular differences between two accelerations

    Args:
        predicted (torch.tensor): model predictions
        labels (torch.tensor): ground truth labels
    Returns:
        [torch.tensor]: mean normalized + contrastive loss
    """
    if predicted.shape[1] != 3:
        raise ValueError(
            "This loss is only available for acceleration-based models.")

    if predicted.shape[0] % 2 > 0:
        raise ValueError("This loss requires even batch size.")

    c = torch.sum(torch.mul(labels.view(-1), predicted.view(-1))) / \
        torch.sum(torch.pow(predicted.view(-1), 2))

    normalized = torch.mean(
        torch.pow(torch.sub(labels.view(-1), c*predicted.view(-1)), 2))

    contrastive = torch.mean(_angular_difference(
        predicted[::2], predicted[1::2], labels[::2], labels[1::2]))

    return normalized + contrastive


def normalized_loss(predicted, labels):
    """Computes the minimal L2 loss between labels and predicted for some scaling parameter c

    Args:
        predicted (torch.tensor): model predictions
        labels (torch.tensor): ground truth labels
    Returns:
        [torch.tensor]: mean normalized + contrastive loss
    """
    c = torch.sum(torch.mul(labels, predicted)) / \
        torch.sum(torch.pow(predicted, 2))
    return torch.sum(torch.pow(torch.sub(labels, c*predicted), 2)) / len(labels)


def normalized_L1_loss(predicted, labels):
    """Computes the minimal L1 loss between labels and predicted for some scaling parameter c

    Args:
        predicted (torch.tensor): model predictions
        labels (torch.tensor): ground truth labels
    Returns:
        [torch.tensor]: mean normalized + contrastive loss
    """
    c = torch.sum(torch.mul(labels, predicted)) / \
        torch.sum(torch.pow(predicted, 2))
    return torch.sum(torch.abs(torch.sub(labels, c*predicted))) / len(labels)


def mse_loss(predicted, labels):
    """Computes the mean squared error between labels and predictions

    Args:
        predicted (torch.tensor): model predictions
        labels (torch.tensor): ground truth labels
    Returns:
        [torch.tensor]: MSE loss
    """
    return torch.nn.MSELoss()(predicted, labels)
