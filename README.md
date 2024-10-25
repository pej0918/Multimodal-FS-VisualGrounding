# Multimodal Few-shot Visual Grounding without Fine-tuning

We propose a **Multimodal Few-shot Visual Grounding** model architecture that eliminates the need for fine-tuning. By enhancing the **Dynamic MDETR** model with **multimodal prompts**, **cross-attention fusion**, and **contrastive loss**, the proposed approach improves visual grounding performance, especially in few-shot scenarios.

## Model Architecture Overview

![Architecture](./images/model.jpg)

Traditional visual grounding models often require large datasets and fine-tuning for new classes, which poses limitations in few-shot learning situations. Our model tackles these challenges by introducing **multimodal prompts** that combine image and text features with **learnable embeddings**, which are then processed through a **cross-attention fusion module**. This mechanism strengthens the interaction between image and text, ensuring better integration of multimodal data. Additionally, **contrastive learning** is employed to enhance class differentiation by maximizing inter-class variability and minimizing intra-class similarity, leading to better generalization for unseen classes. Together, these improvements allow the model to effectively generalize to new tasks without the need for extensive fine-tuning

## Experiment Results

### 1. Environment Setup
```bash
conda create -n dynamic-mdetr python=3.10
conda activate dynamic-mdetr
bash install.txt
```

### 2. Dataset Preparation
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for details on dataset preparation and pretrained checkpoints.

### 3. Training the Model
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --model_type ResNet --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --pretrained_model ./checkpoints/best_checkpoint.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --clip_max_norm 1.0
```

### 4. Evaluation
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --model_type ResNet --batch_size 16 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --eval_model outputs/refcocog_gsplit_r50/best_checkpoint.pth --eval_set val
```

### 5. Results on RefCOCOg
This experiment aimed to evaluate the impact of incorporating templates into the visual grounding process. By comparing our model’s performance on the RefCOCOg dataset with and without templates, we demonstrate **the effectiveness of template-based multimodal prompts and learnable embeddings** in improving accuracy.

| Methods                          | Backbone  | Support Set | Accuracy |
|-----------------------------------|-----------|-------------|----------|
| TransVG                          | ResNet-101| No          | 67.02%   |
| GroundVLP                        | Vin-VL    | No          | 74.73%   |
| Dynamic MDETR                    | ResNet-50 | No          | 69.43%   |
| **Dynamic MDETR + FS-learnable embedding (ours)** | ResNet-50 | Yes        | **83.6%** |

Our model outperformed other baseline models like **TransVG** and **GroundVLP**, achieving **83.6% accuracy**, a significant improvement over other methods without fine-tuning. The integration of **learnable embeddings** and **multimodal prompts** enabled richer visual and textual feature learning, thereby improving grounding precision.

### 6. Results on Unseen Classes

This experiment focused on testing **the model’s generalization capabilities on unseen classes**. By incorporating **Fusion Module** and **Contrastive Loss**, we aimed to evaluate whether the model could effectively adapt to novel classes with minimal data in few-shot visual grounding tasks.
| Methods                       | Backbone  | Acc@50 | AP@50   |
|--------------------------------|-----------|----------|------|
| Ours                           | ResNet-50 | 0.30     | 0.53 |
| Ours + Fusion Module (Fu)      | ResNet-50 | 0.39 (+0.09) | 0.58 (+0.05) |
| Ours + Contrastive Loss (CI)   | ResNet-50 | 0.38 (+0.08) | 0.60 (+0.07) |
| Ours + Fu + CI                 | ResNet-50 | **0.39** (+0.09) | **0.60** (+0.07) |

In the second experiment, we evaluated few-shot visual grounding performance on unseen data. The inclusion of the **Fusion Module** and **Contrastive Loss** significantly improved accuracy and AP, demonstrating the model's capability to generalize effectively without fine-tuning.

### 7. Visual Results:

Below are the visualization results showing the model's predictions and the ground truth for few-shot visual grounding tasks. They demonstrate the effectiveness of our Multimodal Few-shot Visual Grounding model in accurately localizing objects, including the Hachuping character, which further validates the model’s ability to generalize across diverse and unseen data. 
<p align="center">
  <img src="./images/visualization1.jpg" alt="Visualization 1" width="250"/>
  <img src="./images/visualization2.jpg" alt="Visualization 2" width="250"/>
  <img src="./images/visualization3.jpg" alt="Visualization 2" width="250"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/79320f85-cfeb-4e90-b019-1211dad9ce98" alt="Visualization_ha1" width="300"/>
  <img src="https://github.com/user-attachments/assets/70e5b5a6-11d5-46a7-9a3c-08243f8626b7"  alt="Visualization_ha2" width="300"/>
</p>


## Conclusion

Our **Multimodal Few-shot Visual Grounding** model, without the need for fine-tuning, leverages **multimodal prompts**, **cross-attention**, and **contrastive learning** to achieve state-of-the-art performance in visual grounding tasks. The experimental results confirm the effectiveness of our approach in enhancing generalization and improving performance on unseen classes.

## References

```
@InProceedings{Kamath_2021_CVPR,
    author    = {Kamath, Aishwarya and Singh, Mannat and LeCun, Yann and Carion, Nicolas},
    title     = {Dynamic DETR: Few-Shot Detection Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {9747-9756}
}
```
