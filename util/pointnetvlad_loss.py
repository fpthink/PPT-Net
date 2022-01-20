import numpy as np
import math
import torch

def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos

def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False, soft_margin=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies)** 2).sum(2)
    if soft_margin:
        loss = loss.clamp(max=88)
        loss = torch.log(1 + torch.exp(loss))   # softplus
    else:
        loss = loss.clamp(min=0.0)              # hinge  function
    if lazy:                                    # lazy = true
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.mean(1)
    if ignore_zero_loss:                        # false
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies)** 2).sum(2)
    if soft_margin:
        second_loss = second_loss.clamp(max=88)
        second_loss = torch.log(1 + torch.exp(second_loss))
    else:
        second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.mean(1)
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    
    return total_loss

def contrastive_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=True, ignore_zero_loss=False, soft_margin=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)

    negative = ((neg_vecs - query_copies)** 2).sum(2)  # [B, num_neg]
    min_neg = negative.min(1)[0]
    mask = min_neg < positive
    loss1 = loss2 = 0
    if mask.sum() != 0:
        loss1 = m1 + positive[mask].detach() - min_neg[mask]
        loss1 = loss1.clamp(min=0.0).sum()
    mask = ~mask
    if mask.sum() != 0:
        loss2 = m1 + positive[mask] - min_neg[mask]
        loss2 = loss2.clamp(min=0.0).sum()
    triplet_loss = (loss1+loss2)/batch

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies)** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)

    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.mean(1)
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    
    return total_loss


def hphn_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)   # [B]
    min_neg, max_neg = best_pos_distance(q_vec, neg_vecs)   # [B]
    min_other_neg, max_other_neg = best_pos_distance(other_neg, neg_vecs)   # [B]
    
    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    
    hard_neg = torch.stack([min_neg,min_other_neg], dim=1).min(dim=1, keepdim=False)[0]
    hphn_quadruplet = m1 + max_pos - hard_neg
    hphn_quadruplet = hphn_quadruplet.clamp(min=0.0)
    
    return hphn_quadruplet.mean()

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=1,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances.squeeze(), indices.squeeze()

def chamfer_loss(pc1, pc2):
    queries_tensor = torch.from_numpy(pc1[0]).float()
    positives_tensor = torch.from_numpy(pc1[1]).float()
    negatives_tensor = torch.from_numpy(pc1[2]).float()
    other_neg_tensor = torch.from_numpy(pc1[3]).float()
    feed_tensor = torch.cat((queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 4096, 3))
    feed_tensor = feed_tensor.transpose(2,1).contiguous().to(pc2)
    res_tensor = pc2
    dist1, _ = nearest_neighbor(feed_tensor, res_tensor)
    dist2, _ = nearest_neighbor(res_tensor, feed_tensor)
    loss = torch.mean(-dist1) + torch.mean(-dist2)
    return loss
