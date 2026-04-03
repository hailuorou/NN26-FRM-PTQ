<div align="center">

# Feature Relationship Matching Enhanced Low-Bit Post-Training Quantization for Large Language Models

**Chao Zeng**, **[Jiaqi Zhao](https://scholar.google.com/citations?user=lrJ0VWYAAAAJ&hl=zh-CN&oi=sra)**, **[Miao Zhang](https://scholar.google.com/citations?user=XdUDc34AAAAJ&hl=zh-CN&oi=sra)**\*, **[Li Wang](https://scholar.google.com/citations?user=YXJrMkYAAAAJ&hl=zh-CN&oi=sra)**, **Weili Guan**, **[Liqiang Nie](https://scholar.google.com/citations?user=yywVMhUAAAAJ&hl=zh-CN&oi=ao)**  
Harbin Institute of Technology, Shenzhen  
\* Corresponding author

---

[![NN](https://img.shields.io/badge/NN-2026-blue.svg)](https://www.sciencedirect.com/science/article/pii/S089360802600081X)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
<img src="https://img.shields.io/badge/python-≥3.11-blue?style=flat-square" alt="Python">
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

---
Post-Training Quantization (PTQ) has emerged as an effective approach to reduce memory and computational demands during LLMs inference. However, existing PTQ methods are highly sensitive to ultra-low-bit quantization with significant performance loss, which is further exacerbated by recently released advanced models like LLaMA-3 and LLaMA-3.1. To address this challenge, we propose a novel PTQ framework, termed **FRM-PTQ**, by introducing feature relationship matching. This approach integrates token-level relationship modeling and structure-level distribution alignment based on the intra-block self-distillation framework to effectively mitigate significant performance degradation caused by low-bit quantization. Unlike conventional MSE loss methods, which focus solely on point-to-point discrepancies, feature relationship matching captures feature representations in high-dimensional spaces to effectively bridge the representation gap between quantized and full-precision blocks. Additionally, we propose a multi-granularity per-group quantization technique featuring a customized kernel, designed based on the quantization sensitivity of decoder block, to further relieve the quantization performance degradation. Extensive experimental results demonstrate that our method achieves outstanding performance in the W4A4 low-bit scenario, maintaining near full-precision accuracy while delivering a 2 $\times$ throughput improvement and a 3.17 $\times$ memory reduction. This advantage is particularly evident in the latest models such as LLaMA-3, LLaMA-3.1 and Qwen2.5 models, as well as in the W3A3 extreme low-bit scenarios. 

## Usage
We provide full script to run FRM-PTQ. We use LLaMA-2-7B as an example here. You can download the model weights of [LLaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) from [Huggingface](https://huggingface.co/).
1. Install Package
```
conda create -n frm python=3.11.0 -y
conda activate frm
pip install --upgrade pip  
pip install -r requirements.txt
```

2. Get Sensitive Groups:

```
CUDA_VISIBLE_DEVICES=7 python calculate_Kurtosis_llama.py /path/Llama-2-7b
```

3. model quantization
```
# W4A16
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

# W4A4 
CUDA_VISIBLE_DEVICES=7 python main.py \
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

```


## Reproduce Our results 
You can download our pre-released model weights: the LLaMA-2-13B quantized to W2A16, and the LLaMA-3-8B quantized to W3A3 using FRM-PTQ. Then, use the code in `runing_quantized_w2a16_llama_2_13b.ipynb` and `runing_quantized_w3a3_llama_3_8b.ipynb` to reproduce the results presented in our paper. More models and code will be released soon.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{zeng2026frm,
  title={FRM-PTQ: Feature Relationship Matching Enhanced Low-Bit Post-Training Quantization for Large Language Models},
  author={Zeng, Chao and Zhang, Miao and Zhao, Jiaqi and Wang, Li and Guan, Weili and Nie, Liqiang},
  journal={Neural Networks},
  pages={108619},
  year={2026},
  publisher={Elsevier}
}
```
## Acknowledgement

This codebase is heavily built upon [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT), [GPTQ](https://github.com/ist-daslab/gptq) and [Atom](https://github.com/efeslab/Atom). We thank the authors for their excellent work and open-source contributions.

---

## License

This project is released under the Apache License 2.0. See [`LICENSE`](./LICENSE) for details.
