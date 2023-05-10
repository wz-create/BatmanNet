# BatmanNet : Bi-branch Masked Graph Transformer Autoencoder for Molecular Representation

# Requirements
* RDKit : Tested on 2019.09.3
* Python : Tested on 3.8.13
* PyTorch : Tested on 1.9.1

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

For the other packages, please refer to the requirements.txt.

# Quick Start

## Pretraining
Pretrain BatmanNet given the unlabelled molecular data.  
**Note:** There are more hyper-parameters which can be tuned during pretraining. Please refer to `add_pretrain_args/` in `BatmanNet/util/parsing.py/`.

```
python preprocess.py  --data_path data/zinc/all.txt \
                      --output_path data/processed 
                      

```
```
python main.py pretrain \
               --data_path data/processed \
               --save_dir model/pretrained_model/ \
               --batch_size 32 \
               --hidden_size 100
               --dropout 0.1 \
               --depth 3 \
               --num_attn_head 2 \
               --num_enc_mt_block 6\
               --num_dec_mt_block 2\
               --epochs 15 \
               --init_lr 0.0002 \
               --max_lr 0.0004 \
               --final_lr 0.0001 \
               --weight_decay 0.0000001 \
               --activation PReLU 
```

## Finetuning

### Finetuning on Molecular Property Prediction Datasets
#### Molecular Feature Extraction
``` bash
python scripts/save_features.py --data_path data/finetune/bbbp.csv \
                                --save_path data/finetune/bbbp.npz \
                                --features_generator rdkit_2d_normalized \
                                --restart 
```
**Note:** There are more hyper-parameters which can be tuned during finetuning. Please refer to `add_finetune_args` in`BatmanNet/util/parsing.py` .

```
python main.py finetune --data_path data/finetune/bbbp.csv \
                        --features_path data/finetune/bbbp.npz \
                        --save_dir model/finetune/bbbp/ \
                        --checkpoint_path pretrained_model/model/model.ep15 \
                        --dataset_type classification \
                        --split_type scaffold_balanced \
                        --ensemble_size 1 \
                        --num_folds 3 \
                        --ffn_hidden_size 200 \
                        --ffn_num_layer 2\
                        --batch_size 32 \
                        --epochs 100 \
                        --init_lr 0.00015
```
The final finetuned model is stored in `model/bbbp` and will be used in the subsequent prediction and evaluation tasks.  

### Finetuning on DDI Prediction Datasets

```
python DDI/finetune_snap.py --parser_name ddi \
                            --data_path data/biosnap/raw/all.csv \
                            --save_dir model/biosnap/ \
                            --checkpoint_path pretrained_model/model/model.ep15 \
                            --dataset biosnap \
                            --ffn_hidden_size 200 \
                            --ffn_num_layer 2\
                            --batch_size 32 \
                            --epochs 100 \
                            --init_lr 0.00015
```
The final finetuned model is stored in `DDI/runs`


### Finetuning on DTI Prediction Datasets
```
python DTI/cross_validate_human.py  --parser_name dti \
                                    --data_path data/human/raw/data.txt \
                                    --save_dir model/human/ \
                                    --checkpoint_path pretrained_model/model/model.ep15 \
                                    --dataset human \
                                    --model human \
                                    --ffn_hidden_size 200 \
                                    --ffn_num_layer 2 \
                                    --num_folds 3 \
                                    --num_iters 3 \
                                    --batch_size 8 \
                                    --epochs 30 \
                                    --lr 0.00005 \
                                    --hid 32 \
                                    --heads 4 \
                                    --deep 1 \
                                    --dropout 0.2 
                                    
```
The final finetuned model is stored in `DTI/test`