# Auto-Diagnosis-by-RL-and-Classification

How to use our code:
1. The proposed MedlinePlus dataset can be found in ./environment/medlineplus.jsonCancel changes
2. To run test the method on medlineplus, 
   1. cd ./medlineplus_code
   2. CUDA_VISIBLE_DEVICES=0 python3 main.py -train -trail 1
3. To test the method on symcat disease sets,
   1. cd ./symcat_code
   2. CUDA_VISIBLE_DEVICES=0 python3 main.py -train -train 1 -dataset 200 (300/400/common)
 
#Citation

The citation for our paper is:
```bibtex
@misc{
  efficientautodiag,
  doi = {10.48550/ARXIV.2112.00733},
  url = {https://arxiv.org/abs/2112.00733},
  author = {Yuan, Hongyi and Yu, Sheng},
  title = {Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
