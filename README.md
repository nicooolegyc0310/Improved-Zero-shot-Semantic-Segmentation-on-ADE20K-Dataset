Read Me

Mainly based ZegCLIP open source：

（1）ADE20K dataloader：configs/_base_/datasets/dataloader/ade20.py
<img width="146" alt="image" src="https://github.com/nicooolegyc0310/Improved-Zero-shot-Semantic-Segmentation-on-ADE20K-Dataset/assets/168481450/16d4a661-a069-4ede-b204-a6f061baf0db">


Get classes names embeddings of ADE20K from CLIP encoder:
configs/_base_/datasets/text_embedding/get_embedding.ipynb

<img width="452" alt="image" src="https://github.com/nicooolegyc0310/Improved-Zero-shot-Semantic-Segmentation-on-ADE20K-Dataset/assets/168481450/af4934bf-8f79-488f-b558-ea56f7b9aef0">

 
The results is saved at configs/_base_/datasets/text_embedding/ade_multi.npy

(2) Running config on ADE20K:

<img width="400" alt="image" src="https://github.com/nicooolegyc0310/Improved-Zero-shot-Semantic-Segmentation-on-ADE20K-Dataset/assets/168481450/43560b0e-ed4f-4cbe-8870-ea417aaf6a45">

 
1.	Baseline (ZegCLIP method): configs/ade20k/vpt_seg_zero_vit-b_512x512_160k_12_100_multi.py
2.	Improved (this project): configs/ade20k/lora_seg_zero_vit-b_512x512_160k_multi.py

(3) Method update:

<img width="452" alt="image" src="https://github.com/nicooolegyc0310/Improved-Zero-shot-Semantic-Segmentation-on-ADE20K-Dataset/assets/168481450/4b68199f-d67a-4d5c-b206-53cf33c9f60d">


1.	Main Class: models/backbone/img_encoder.py
   
<img width="366" alt="image" src="https://github.com/nicooolegyc0310/Improved-Zero-shot-Semantic-Segmentation-on-ADE20K-Dataset/assets/168481450/100ef859-4323-4cae-b61c-d2c8ecea2632">

 
2.	The reliable Classes of LoRACLIPVisionTransformer are written in:
models/backbone/loralib.py
models/backbone/utils.py


(4) Follow the README.md to create conda environment, and then run:
Training:
1.	bash dist_train.sh configs/ade20k/vpt_seg_zero_vit-b_512x512_160k_12_100_multi.py ./checkpoint/baseline 
2.	bash dist_train.sh configs/ade20k/lora_seg_zero_vit-b_512x512_160k_multi.py ./checkpoint/improved

Testing:
3.	python test.py configs/ade20k/vpt_seg_zero_vit-b_512x512_160k_12_100_multi.py ./checkpoint/baseline/latest.pth –eval=mIoU

4.	python test.py configs/ade20k/ lora_seg_zero_vit-b_512x512_160k_multi.py ./checkpoint/baseline/latest.pth –eval=mIoU
