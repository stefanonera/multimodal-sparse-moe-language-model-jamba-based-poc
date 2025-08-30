# Multimodal Sparse MoE Language Model - Jamba-Based POC

A proof-of-concept implementation of a multimodal classifier using sparse Mixture of Experts (MoE) architecture with top-2 routing and frozen encoders.

## Overview

This POC demonstrates a minimal multimodal pipeline featuring:
- **Sparse MoE Block**: Top-2 routing with SwiGLU experts
- **Frozen Encoders**: Minimal computational overhead with pre-trained components
- **Multimodal Fusion**: Unified processing across different input modalities

## Supported Modalities

- **Image + Text**: CIFAR-10 dataset
- **Text Only**: AG News classification
- **Video + Text**: Mini-UCF (subset of UCF101 Actions from Kaggle)

## Key Contributions

1. **Minimal Architecture**: Frozen encoders + fusion layer + MoE block with SwiGLU experts
2. **Unified Training Pipeline**: Single framework supporting three different modalities
3. **Routing Analytics**: Clear metrics and heatmaps for expert utilization analysis
4. **Ablation Studies**: Small grid of scripted experimental runs

## Goals

- Demonstrate **routing correctness** and proper expert utilization
- Provide **instrumentation** for MoE behavior analysis
- Show **minimal performance lift** compared to dense alternatives
- **Not aimed at SOTA** - focused on architectural validation

## Scope Limitations

- Brief training cycles (proof-of-concept only)
- Classification tasks only
- Text encoder is a frozen stub implementation
- Video dataset is intentionally small (mini-UCF subset)

## Purpose

This project serves as a foundational exploration of multimodal sparse MoE architectures, providing insights into routing behavior and expert specialization across different input modalities while maintaining computational efficiency through frozen encoder components.
