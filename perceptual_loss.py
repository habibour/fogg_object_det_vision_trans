"""
Perceptual Loss for PL-RT-DETR
Based on: "Weather-Aware Object Detection Transformer for Domain Adaptation"

Implements perceptual loss to preserve high-level semantic features
between clean (teacher) and foggy (student) image representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict, Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pretrained CNN features.
    
    Formula from paper: L_perc = Σ λ_l * |f^(l)(I_s) - f^(l)(I_t)|^2
    where f^(l) are features from layer l of a pretrained network.
    """
    
    def __init__(
        self,
        network: str = 'vgg16',
        layers: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            network: 'vgg16', 'vgg19', or 'resnet50'
            layers: List of layer names to extract features from
            weights: Weights (λ_l) for each layer's contribution
            reduction: 'mean' or 'sum' for loss reduction
        """
        super().__init__()
        
        self.reduction = reduction
        
        # Load pretrained network
        if network == 'vgg16':
            pretrained = models.vgg16(pretrained=True)
            self.feature_extractor = pretrained.features
            if layers is None:
                # Use intermediate conv layers
                layers = ['4', '9', '16', '23', '30']  # After each maxpool
            if weights is None:
                weights = [1.0, 1.0, 1.0, 1.0, 1.0]
                
        elif network == 'vgg19':
            pretrained = models.vgg19(pretrained=True)
            self.feature_extractor = pretrained.features
            if layers is None:
                layers = ['4', '9', '18', '27', '36']
            if weights is None:
                weights = [1.0, 1.0, 1.0, 1.0, 1.0]
                
        elif network == 'resnet50':
            pretrained = models.resnet50(pretrained=True)
            # For ResNet, we'll extract from different blocks
            if layers is None:
                layers = ['layer1', 'layer2', 'layer3', 'layer4']
            if weights is None:
                weights = [1.0, 1.0, 1.0, 1.0]
            self.is_resnet = True
        else:
            raise ValueError(f"Unsupported network: {network}")
        
        self.network_name = network
        self.layers = layers
        self.weights = weights
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_extractor.eval()
        
        # Normalization (ImageNet stats)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def extract_features_vgg(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from VGG network."""
        features = []
        for name, module in self.feature_extractor._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features
    
    def extract_features_resnet(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from ResNet network."""
        features = []
        
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            x = getattr(self.feature_extractor, layer_name)(x)
            if layer_name in self.layers:
                features.append(x)
        
        return features
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Compute perceptual loss between source and target.
        
        Args:
            source: Source image (e.g., clean image features) [B, C, H, W]
            target: Target image (e.g., foggy image features) [B, C, H, W]
            return_features: If True, also return extracted features
            
        Returns:
            loss: Perceptual loss value
            features (optional): Dict of extracted features
        """
        # Normalize inputs
        source_norm = self.normalize(source)
        target_norm = self.normalize(target)
        
        # Extract features
        with torch.no_grad() if not self.training else torch.enable_grad():
            if 'resnet' in self.network_name:
                source_features = self.extract_features_resnet(source_norm)
                target_features = self.extract_features_resnet(target_norm)
            else:
                source_features = self.extract_features_vgg(source_norm)
                target_features = self.extract_features_vgg(target_norm)
        
        # Compute loss at each layer
        loss = 0.0
        for i, (sf, tf, weight) in enumerate(zip(source_features, target_features, self.weights)):
            # L2 loss between features
            layer_loss = F.mse_loss(sf, tf, reduction=self.reduction)
            loss += weight * layer_loss
        
        if return_features:
            return loss, {
                'source_features': source_features,
                'target_features': target_features
            }
        
        return loss


class FeaturePerceptualLoss(nn.Module):
    """
    Perceptual loss computed on RT-DETR backbone features.
    
    This version compares the intermediate features from the teacher
    and student RT-DETR backbones directly, rather than using a separate
    pretrained network.
    """
    
    def __init__(
        self,
        feature_layers: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            feature_layers: Indices of feature layers to compare
            weights: Weights for each layer's contribution
            reduction: 'mean' or 'sum' for loss reduction
        """
        super().__init__()
        
        self.feature_layers = feature_layers or [0, 1, 2, 3]
        self.weights = weights or [1.0] * len(self.feature_layers)
        self.reduction = reduction
        
    def forward(
        self,
        teacher_features: List[torch.Tensor],
        student_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute perceptual loss between teacher and student features.
        
        Args:
            teacher_features: List of feature maps from teacher backbone
            student_features: List of feature maps from student backbone
            
        Returns:
            loss: Perceptual loss value
        """
        loss = 0.0
        
        for idx, weight in zip(self.feature_layers, self.weights):
            if idx < len(teacher_features) and idx < len(student_features):
                teacher_feat = teacher_features[idx]
                student_feat = student_features[idx]
                
                # Ensure same spatial dimensions
                if teacher_feat.shape != student_feat.shape:
                    student_feat = F.interpolate(
                        student_feat,
                        size=teacher_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute MSE loss
                layer_loss = F.mse_loss(teacher_feat, student_feat, reduction=self.reduction)
                loss += weight * layer_loss
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for PL-RT-DETR training.
    
    L_total = L_detection + α * L_perceptual
    
    where L_detection is the original RT-DETR loss (classification + bbox regression)
    and L_perceptual is the perceptual loss for domain adaptation.
    """
    
    def __init__(
        self,
        perceptual_weight: float = 1.0,
        use_image_perceptual: bool = True,
        use_feature_perceptual: bool = True
    ):
        """
        Args:
            perceptual_weight: Weight α for perceptual loss
            use_image_perceptual: Use image-level perceptual loss (VGG/ResNet)
            use_feature_perceptual: Use feature-level perceptual loss (backbone features)
        """
        super().__init__()
        
        self.perceptual_weight = perceptual_weight
        self.use_image_perceptual = use_image_perceptual
        self.use_feature_perceptual = use_feature_perceptual
        
        if use_image_perceptual:
            self.image_perceptual = PerceptualLoss(network='vgg16')
        
        if use_feature_perceptual:
            self.feature_perceptual = FeaturePerceptualLoss()
    
    def forward(
        self,
        detection_loss: torch.Tensor,
        clean_images: Optional[torch.Tensor] = None,
        foggy_images: Optional[torch.Tensor] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
        student_features: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            detection_loss: RT-DETR detection loss
            clean_images: Clean images (for teacher)
            foggy_images: Foggy images (for student)
            teacher_features: Features from teacher backbone
            student_features: Features from student backbone
            
        Returns:
            Dictionary with 'total_loss', 'detection_loss', 'perceptual_loss'
        """
        losses = {'detection_loss': detection_loss}
        
        perceptual_loss = 0.0
        
        # Image-level perceptual loss
        if self.use_image_perceptual and clean_images is not None and foggy_images is not None:
            img_perc_loss = self.image_perceptual(clean_images, foggy_images)
            perceptual_loss += img_perc_loss
            losses['image_perceptual_loss'] = img_perc_loss
        
        # Feature-level perceptual loss
        if self.use_feature_perceptual and teacher_features is not None and student_features is not None:
            feat_perc_loss = self.feature_perceptual(teacher_features, student_features)
            perceptual_loss += feat_perc_loss
            losses['feature_perceptual_loss'] = feat_perc_loss
        
        losses['perceptual_loss'] = perceptual_loss
        losses['total_loss'] = detection_loss + self.perceptual_weight * perceptual_loss
        
        return losses


if __name__ == '__main__':
    # Test perceptual loss
    print("Testing Perceptual Loss Module...")
    print("=" * 60)
    
    # Create dummy images
    batch_size = 2
    clean_images = torch.randn(batch_size, 3, 640, 640)
    foggy_images = torch.randn(batch_size, 3, 640, 640)
    
    # Test image perceptual loss
    print("\n1. Testing Image Perceptual Loss (VGG16)...")
    perc_loss = PerceptualLoss(network='vgg16')
    loss = perc_loss(clean_images, foggy_images)
    print(f"   Perceptual loss: {loss.item():.4f}")
    
    # Test feature perceptual loss
    print("\n2. Testing Feature Perceptual Loss...")
    teacher_features = [
        torch.randn(batch_size, 256, 160, 160),
        torch.randn(batch_size, 512, 80, 80),
        torch.randn(batch_size, 1024, 40, 40),
        torch.randn(batch_size, 2048, 20, 20)
    ]
    student_features = [
        torch.randn(batch_size, 256, 160, 160),
        torch.randn(batch_size, 512, 80, 80),
        torch.randn(batch_size, 1024, 40, 40),
        torch.randn(batch_size, 2048, 20, 20)
    ]
    
    feat_perc_loss = FeaturePerceptualLoss()
    loss = feat_perc_loss(teacher_features, student_features)
    print(f"   Feature perceptual loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n3. Testing Combined Loss...")
    detection_loss = torch.tensor(1.5)
    combined_loss = CombinedLoss(perceptual_weight=1.0)
    
    losses = combined_loss(
        detection_loss=detection_loss,
        clean_images=clean_images,
        foggy_images=foggy_images,
        teacher_features=teacher_features,
        student_features=student_features
    )
    
    print(f"   Detection loss: {losses['detection_loss'].item():.4f}")
    print(f"   Perceptual loss: {losses['perceptual_loss'].item():.4f}")
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Perceptual Loss Module test complete!")
