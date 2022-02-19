# Example:
# gpu=0 lr_mul=2 scale_emb_or_prj=prj bash train_multi30k_de_en.sh
# The better setting is:
# gpu=0 lr_mul=0.5 scale_emb_or_prj=emb bash train_multi30k_de_en.sh

python train.py -data_pkl m30k_deen_shr.pkl  -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400 -no_cuda -d_k 32 -d_v 32 -d_model 32 -d_inner 32 -n_layer 2 -n_head 10

exit 0
CUDA_VISIBLE_DEVICES=${gpu} python train.py \
-data_pkl multi30k_de_en.pkl \
-label_smoothing \
-proj_share_weight \
-scale_emb_or_prj ${scale_emb_or_prj} \
-lr_mul ${lr_mul} \
-b 256 \
-warmup 4000 \
-epoch 200 \
-seed 1 \
-output_dir output/lr_mul_${lr_mul}-scale_${scale_emb_or_prj} \
-use_tb
