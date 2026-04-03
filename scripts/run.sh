# get multi-group quantization groups
CUDA_VISIBLE_DEVICES=7 python calculate_Kurtosis_llama.py /root/dataset/Llama-2-7b

# quantization
CUDA_VISIBLE_DEVICES=7 python main.py \
--model /root/dataset/Llama-2-7b \
--output_dir ./output/Llama-2-7b-w4g128 \
--net Llama-2 \
--wbits 4 \
--group_size 128 \
--quant_lr 2e-5 \
--calib_dataset wikitext2 \
--real_quant \
--epochs 1 \
--seed 42 \
--train_size 128 \
--batch_size 1 \
--sensitive_group 6 3 2 22 30 31 1 0 \
--robust_group 25 27 23 28 18 15 \
--eval_ppl \
--save_quant_dir ./output/Llama-2-7b-w4g128

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /root/dataset/Llama-2-7b \
--output_dir ./output/Llama-2-7b-w4a4g128 \
--net Llama-2 \
--wbits 4 \
--abits 4 \
--use_act_quant \
--group_size 128 \
--quant_lr 2e-5 \
--calib_dataset wikitext2 \
--real_quant \
--epochs 1 \
--seed 42 \
--train_size 128 \
--batch_size 1 \
--sensitive_group 6 3 2 22 30 31 1 0 \
--robust_group 25 27 23 28 18 15 \
--eval_ppl \
--save_quant_dir ./output/Llama-2-7b-w4a4g128
