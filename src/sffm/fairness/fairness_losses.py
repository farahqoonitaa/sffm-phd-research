"""Fairness Loss Functions for Sequential Financial Foundation Model.

Implements 6 fairness algorithms for Phase 3 experimentation:
1. Demographic Parity
2. Equalized Odds
3. Calibration Fairness
4. Counterfactual Fairness
5. Peer-Induced Fairness (RECOMMENDED)
6. IBM 360 Fairness (NOVEL - Comprehensive Multi-Metric Approach)

Author: Farah Qoonita Syuhaila
Email: farahqoonita2@gmail.com
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class FairnessLosses(nn.Module):
    """Centralised fairness loss computation for SFFM.
    
    Supports 6 fairness algorithms:
    1. demographic_parity - Equal average predictions across groups
    2. equalized_odds - Equal error rates across groups
    3. calibration - Equal uncertainty across groups
    4. counterfactual - Causal fairness via counterfactual inference
    5. peer_induced - Peer-induced fairness (RECOMMENDED)
    6. ibm_360 - IBM 360 Fairness (comprehensive multi-metric)
    """
    
    def __init__(self, method: str = 'peer_induced', lambda_fair: float = 0.3):
        super().__init__()
        self.method = method
        self.lambda_fair = lambda_fair
        
        if method not in ['demographic_parity', 'equalized_odds', 'calibration',
                         'counterfactual', 'peer_induced', 'ibm_360']:
            raise ValueError(f"Unknown fairness method: {method}")
    
    # ==================== 1. DEMOGRAPHIC PARITY ====================
    
    def demographic_parity_loss(self, pred: torch.Tensor, 
                               protected_attr: torch.Tensor) -> torch.Tensor:
        """Enforce equal average predictions across groups.
        
        Loss = |E[pred | Group A] - E[pred | Group B]|
        """
        mask_a = protected_attr == 0
        mask_b = protected_attr == 1
        
        mean_a = pred[mask_a].mean() if mask_a.sum() > 0 else 0
        mean_b = pred[mask_b].mean() if mask_b.sum() > 0 else 0
        
        loss = (mean_a - mean_b) ** 2
        return loss
    
    # ==================== 2. EQUALIZED ODDS ====================
    
    def equalized_odds_loss(self, pred: torch.Tensor, 
                           target: torch.Tensor,
                           protected_attr: torch.Tensor,
                           threshold: float = 0.5) -> torch.Tensor:
        """Enforce equal TPR and FPR across groups."""
        pred_binary = (pred > threshold).float()
        loss = torch.tensor(0.0, device=pred.device)
        
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            pred_g = pred_binary[mask]
            target_g = target[mask]
            
            tp = ((pred_g == 1) & (target_g == 1)).sum().float()
            p = (target_g == 1).sum().float()
            tpr = tp / (p + 1e-8)
            
            fp = ((pred_g == 1) & (target_g == 0)).sum().float()
            n = (target_g == 0).sum().float()
            fpr = fp / (n + 1e-8)
            
            if group == 0:
                tpr_a, fpr_a = tpr, fpr
            else:
                tpr_b, fpr_b = tpr, fpr
        
        loss = (tpr_a - tpr_b) ** 2 + (fpr_a - fpr_b) ** 2
        return loss
    
    # ==================== 3. CALIBRATION FAIRNESS ====================
    
    def calibration_fairness_loss(self, pred: torch.Tensor,
                                 target: torch.Tensor,
                                 protected_attr: torch.Tensor,
                                 num_bins: int = 10) -> torch.Tensor:
        """Enforce equal calibration (ECE) across groups."""
        loss = torch.tensor(0.0, device=pred.device)
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=pred.device)
        
        ece_by_group = []
        
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() == 0:
                continue
            
            pred_g = pred[mask]
            target_g = target[mask]
            
            ece = torch.tensor(0.0, device=pred.device)
            for i in range(num_bins):
                bin_mask = (pred_g >= bin_edges[i]) & (pred_g < bin_edges[i + 1])
                if bin_mask.sum() > 0:
                    acc = target_g[bin_mask].mean()
                    conf = pred_g[bin_mask].mean()
                    ece += torch.abs(acc - conf)
            
            ece_by_group.append(ece / num_bins)
        
        if len(ece_by_group) == 2:
            loss = (ece_by_group[0] - ece_by_group[1]) ** 2
        
        return loss
    
    # ==================== 4. COUNTERFACTUAL FAIRNESS ====================
    
    def counterfactual_fairness_loss(self, pred: torch.Tensor,
                                    features: torch.Tensor,
                                    protected_attr: torch.Tensor,
                                    k_neighbors: int = 5) -> torch.Tensor:
        """Penalise decisions that would change if protected attr flipped."""
        loss = torch.tensor(0.0, device=pred.device)
        batch_size = len(pred)
        
        for i in range(min(batch_size, 100)):
            other_group = 1 - protected_attr[i]
            mask = protected_attr == other_group
            
            if mask.sum() == 0:
                continue
            
            other_features = features[mask]
            other_pred = pred[mask]
            
            distances = ((other_features - features[i]) ** 2).sum(dim=1)
            k = min(k_neighbors, mask.sum())
            nearest_idx = torch.topk(distances, k, largest=False)[1]
            
            counterfactual_pred = other_pred[nearest_idx].mean()
            diff = torch.abs(pred[i] - counterfactual_pred)
            loss += diff ** 2
        
        return loss / min(batch_size, 100)
    
    # ==================== 5. PEER-INDUCED FAIRNESS ====================
    
    def peer_induced_fairness_loss(self, pred: torch.Tensor,
                                  features: torch.Tensor,
                                  protected_attr: torch.Tensor,
                                  k_peers: int = 5) -> torch.Tensor:
        """Peer-induced fairness: penalise discrimination vs. similar peers."""
        loss = torch.tensor(0.0, device=pred.device)
        batch_size = len(pred)
        
        for i in range(min(batch_size, 100)):
            distances = ((features - features[i]) ** 2).sum(dim=1)
            distances[i] = float('inf')
            
            k = min(k_peers, batch_size - 1)
            peer_indices = torch.topk(distances, k, largest=False)[1]
            
            for j in peer_indices:
                if protected_attr[i] != protected_attr[j]:
                    if pred[j] < pred[i]:
                        discrimination = (pred[i] - pred[j]) ** 2
                        loss += discrimination
        
        return loss / (min(batch_size, 100) * k)
    
    # ==================== 6. IBM 360 FAIRNESS (NOVEL) ====================
    
    def ibm_360_fairness_loss(self, pred: torch.Tensor,
                             target: torch.Tensor,
                             protected_attr: torch.Tensor,
                             features: torch.Tensor,
                             std: Optional[torch.Tensor] = None,
                             threshold: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """IBM 360 Fairness: 7-metric comprehensive fairness framework."""
        metrics_dict = {}
        device = pred.device
        batch_size = len(pred)
        
        pred_binary = (pred > threshold).float()
        mask_a = protected_attr == 0
        mask_b = protected_attr == 1
        
        sr_a = pred_binary[mask_a].mean() if mask_a.sum() > 0 else 0.5
        sr_b = pred_binary[mask_b].mean() if mask_b.sum() > 0 else 0.5
        dir_ratio = (sr_b / (sr_a + 1e-8)).clamp(min=0.01, max=100.0)
        dir_penalty = torch.abs(dir_ratio - 1.0)
        metrics_dict['dir'] = dir_ratio.item()
        metrics_dict['dir_penalty'] = dir_penalty.item()
        
        cal_loss = torch.tensor(0.0, device=device)
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() > 0:
                pred_g = pred[mask]
                target_g = target[mask]
                pred_binary_g = (pred_g > threshold).float()
                correct = (pred_binary_g == target_g).float().mean()
                cal_loss += torch.abs(correct - pred_g.mean())
        metrics_dict['calibration_parity'] = (cal_loss / 2).item()
        
        theil_loss = torch.tensor(0.0, device=device)
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() > 0:
                pred_g = torch.clamp(pred[mask], min=1e-8, max=1.0 - 1e-8)
                mean_pred = pred_g.mean()
                theil_g = torch.mean(torch.log(mean_pred / (pred_g + 1e-8)))
                theil_loss += torch.abs(theil_g)
        metrics_dict['theil_index'] = (theil_loss / 2).item()
        
        ind_fair_loss = torch.tensor(0.0, device=device)
        sample_indices = torch.randperm(batch_size)[:min(50, batch_size)]
        for i in sample_indices:
            distances = ((features - features[i]) ** 2).sum(dim=1)
            distances[i] = float('inf')
            k_similar = min(5, batch_size - 1)
            similar_idx = torch.topk(distances, k_similar, largest=False)[1]
            ind_fair_loss += pred[similar_idx].var()
        metrics_dict['individual_fairness'] = (ind_fair_loss / len(sample_indices)).item()
        
        treatment_loss = torch.tensor(0.0, device=device)
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() > 0:
                pred_g = (pred[mask] > threshold).float()
                target_g = target[mask]
                tp = ((pred_g == 1) & (target_g == 1)).sum().float()
                p = (target_g == 1).sum().float()
                tpr = tp / (p + 1e-8)
                tn = ((pred_g == 0) & (target_g == 0)).sum().float()
                n = (target_g == 0).sum().float()
                tnr = tn / (n + 1e-8)
                if group == 0:
                    tpr_a, tnr_a = tpr, tnr
                else:
                    tpr_b, tnr_b = tpr, tnr
        treatment_loss = torch.abs(tpr_a - tpr_b) + torch.abs(tnr_a - tnr_b)
        metrics_dict['treatment_equality'] = treatment_loss.item()
        
        pred_parity_loss = torch.tensor(0.0, device=device)
        for group in [0, 1]:
            mask = protected_attr == group
            if mask.sum() > 0:
                pred_g = (pred[mask] > threshold).float()
                target_g = target[mask]
                tp = ((pred_g == 1) & (target_g == 1)).sum().float()
                fp = ((pred_g == 1) & (target_g == 0)).sum().float()
                precision = tp / (tp + fp + 1e-8)
                if group == 0:
                    prec_a = precision
                else:
                    prec_b = precision
        pred_parity_loss = torch.abs(prec_a - prec_b)
        metrics_dict['predictive_parity'] = pred_parity_loss.item()
        
        uncertainty_fairness = torch.tensor(0.0, device=device)
        if std is not None:
            for group in [0, 1]:
                mask = protected_attr == group
                if mask.sum() > 0:
                    std_g = std[mask].mean()
                    if group == 0:
                        std_a = std_g
                    else:
                        std_b = std_g
            uncertainty_fairness = torch.abs(std_a - std_b)
            metrics_dict['uncertainty_fairness'] = uncertainty_fairness.item()
        
        combined_loss = (
            0.20 * dir_penalty + 0.15 * (cal_loss / 2) + 0.10 * (theil_loss / 2) +
            0.15 * (ind_fair_loss / len(sample_indices)) + 0.20 * treatment_loss +
            0.15 * pred_parity_loss + 0.05 * uncertainty_fairness
        )
        metrics_dict['ibm_360_combined'] = combined_loss.item()
        
        return combined_loss, metrics_dict
    
    # ==================== MASTER LOSS ====================
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
               protected_attr: torch.Tensor, features: torch.Tensor,
               mean: Optional[torch.Tensor] = None,
               std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute fairness loss based on selected method."""
        metrics = {}
        
        if self.method == 'demographic_parity':
            loss = self.demographic_parity_loss(pred, protected_attr)
            metrics['dp_loss'] = loss.item()
        elif self.method == 'equalized_odds':
            loss = self.equalized_odds_loss(pred, target, protected_attr)
            metrics['eo_loss'] = loss.item()
        elif self.method == 'calibration':
            loss = self.calibration_fairness_loss(pred, target, protected_attr)
            metrics['cal_loss'] = loss.item()
        elif self.method == 'counterfactual':
            loss = self.counterfactual_fairness_loss(pred, features, protected_attr)
            metrics['cf_loss'] = loss.item()
        elif self.method == 'peer_induced':
            loss = self.peer_induced_fairness_loss(pred, features, protected_attr)
            metrics['pif_loss'] = loss.item()
        elif self.method == 'ibm_360':
            loss, ibm_metrics = self.ibm_360_fairness_loss(
                pred, target, protected_attr, features, std
            )
            metrics.update(ibm_metrics)
        
        total_loss = self.lambda_fair * loss
        return total_loss, metrics
