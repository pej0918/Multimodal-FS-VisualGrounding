# Multimodal Few-shot Visual Grounding without Fine-tuning

We propose a **Multimodal Few-shot Visual Grounding** model architecture that eliminates the need for fine-tuning. By enhancing the **Dynamic MDETR** model with **multimodal prompts**, **cross-attention fusion**, and **contrastive loss**, the proposed approach improves visual grounding performance, especially in few-shot scenarios.

## 🧠 Model Architecture Overview

![Architecture](./images/model.jpg)

The architecture integrates **Multimodal Prompts** combining text, image, and learnable embeddings, as shown in the above figure. Each template provides visual and textual features, further enhanced by a fusion module. Cross-attention mechanisms applied within the fusion module ensure stronger interactions between the two modalities.

🔥 Key improvements in our model include:

- **Multimodal Prompts**: Combining image and text embeddings with a learnable embedding, enabling the model to better capture context and meaning.
- **Cross-Attention Fusion**: The cross-attention fusion module strengthens interactions between image and text modalities, allowing for better multimodal integration.
- **Contrastive Loss**: By maximizing inter-class differences and minimizing intra-class variations, contrastive loss further refines the grounding results and improves generalization across unseen classes.

## 🧬 Methodology

This approach integrates **multimodal prompts with cross-class templates**, **inter-modal cross-attention**, and **contrastive learning** to create a robust model for few-shot visual grounding. By enabling adaptability and strong generalization capabilities without the need for fine-tuning, our model is well-suited for diverse classes and contexts.

### 1️⃣ **Multimodal Prompt Generation**
![image](https://github.com/user-attachments/assets/b12147fb-13be-42d4-838d-bca314b677cd)

Combines visual and textual features with a Learnable Embedding, incorporating templates from both the target and different classes. This strategy helps the model distinguish between classes effectively, enhancing generalization with minimal data.

### 2️⃣ **Cross-Attention Fusion**
![image](https://github.com/user-attachments/assets/9f8533c3-52cb-45fb-ad21-2657ed625ce9)
Applies bidirectional cross-attention (image-to-text and text-to-image) to strengthen multimodal integration, enabling the model to focus on complementary features across modalities.

### 3️⃣ **Contrastive Learning**
![image](https://github.com/user-attachments/assets/21a80df6-1c38-4800-aec4-c0f8231a2dd5)

Improves class differentiation by maximizing inter-class separation and minimizing intra-class variation, ensuring robust performance even on unseen classes.


## 🚀 Usage

### 1️⃣ Environment Setup
```bash
conda create -n dynamic-mdetr python=3.10
conda activate dynamic-mdetr
bash install.txt
```

### 2️⃣ Dataset Preparation
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for details on dataset preparation and pretrained checkpoints.

### 3️⃣ Training the Model
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --weight_contrast 0.2 --use_cross_attention 1 --contrastive_loss 1 --cropped_templates 0 --category_file_path ./path/to/coco_80.txt --pretrained_model /path/to/pretrained model --model_type ResNet --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/flickr_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --epochs 10 --lr_drop 60  --vl_dec_layers 1 --vl_enc_layers 1 --clip_max_norm 1.0
```

### 4️⃣ Evaluation
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --model_type ResNet --batch_size 16 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --eval_model outputs/refcocog_gsplit_r50/best_checkpoint.pth --eval_set val
```

### 5️⃣ Inference
```bash
!python -m torch.distributed.launch --nproc_per_node=1 --use_env inference.py \
  --model_type ResNet \
  --batch_size 1 \
  --backbone resnet50 \
  --bert_enc_num 12 \
  --detr_enc_num 6 \
  --dataset hachuping \
  --max_query_len 40 \
  --output_dir outputs/refcocog_gsplit_r50/inference \
  --stages 3 \
  --vl_fusion_enc_layers 3 \
  --uniform_learnable True \
  --in_points 36 \
  --lr 1e-4 \
  --different_transformer True \
  --data_root /content/drive/MyDrive/fsod/train \
  --eval_model ./path/to/your model \
  --category_file_path /path/to/your cateogry \
  --num_templates 3 \
  --template_classes 3 \
  --use_cross_attention 1 \
  --cropped_templates 0 \
  --vl_dec_layers 1 \
  --vl_enc_layers 1 \
  --eval_set val
```
## 📈 Results
## Results on RefCOCOg
To evaluate the influence of **templates** and **multimodal prompts**, we conducted experiments on the RefCOCOg dataset. The goal was to analyze how the incorporation of templates impacts the model’s visual grounding performance.

| Methods                          | Backbone  | Support Set | Accuracy |
|-----------------------------------|-----------|-------------|----------|
| TransVG                          | ResNet-101| No          | 67.02%   |
| GroundVLP                        | Vin-VL    | No          | 74.73%   |
| Dynamic MDETR                    | ResNet-50 | No          | 69.43%   |
| **Dynamic MDETR + FS-learnable embedding (ours)** | ResNet-50 | Yes        | **83.6%** |

Our model outperformed other baseline models like **TransVG** and **GroundVLP**, achieving **83.6% accuracy**, a significant improvement over other methods without fine-tuning. The integration of **learnable embeddings** and **multimodal prompts** enabled richer visual and textual feature learning, thereby improving grounding precision.
Our model demonstrates a significant improvement in accuracy, achieving **83.6%**, validating the effectiveness of using **learnable embeddings** and **multimodal prompts**.

### Results on Unseen Classes

This experiment focused on assessing the model’s generalization to unseen classes in a few-shot learning context.

| Methods                       | Backbone  | Acc@50 | AP@50   |
|--------------------------------|-----------|----------|------|
| Ours                           | ResNet-50 | 0.30     | 0.53 |
| Ours + Fusion Module (Fu)      | ResNet-50 | 0.39 (+0.09) | 0.58 (+0.05) |
| Ours + Contrastive Loss (CI)   | ResNet-50 | 0.38 (+0.08) | 0.60 (+0.07) |
| Ours + Fu + CI                 | ResNet-50 | **0.39** (+0.09) | **0.60** (+0.07) |

Incorporating **Fusion Module** and **Contrastive Loss** led to a significant improvement in both accuracy and AP, confirming the model’s ability to generalize without the need for fine-tuning.

## 🖼️ Visual Results:

Below are the visualization results showing the model's predictions and the ground truth for few-shot visual grounding tasks. They demonstrate the effectiveness of our **Multimodal Few-shot Visual Grounding** model in accurately localizing objects, which further validates the model’s ability to generalize across diverse and unseen data. 
<p align="center">
  <img src="./images/visualization1.jpg" alt="Visualization 1" width="250"/>
  <img src="./images/visualization2.jpg" alt="Visualization 2" width="250"/>
  <img src="./images/visualization3.jpg" alt="Visualization 2" width="250"/>
</p>

## 🎯 Conclusion

Our **Multimodal Few-shot Visual Grounding** model, without the need for fine-tuning, leverages **multimodal prompts**, **cross-attention**, and **contrastive learning** to achieve state-of-the-art performance in visual grounding tasks. The experimental results confirm the effectiveness of our approach in enhancing generalization and improving performance on unseen classes.

## 📚 References

```
@InProceedings{Kamath_2021_CVPR,
    author    = {Kamath, Aishwarya and Singh, Mannat and LeCun, Yann and Carion, Nicolas},
    title     = {Dynamic DETR: Few-Shot Detection Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {9747-9756}
}
```
