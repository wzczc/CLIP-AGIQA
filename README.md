# CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP

This is the official repo of our ICPR2024 paper [CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP](https://arxiv.org/abs/2408.15098).

## Intruction

CLIP-AGIQA is a CLIP-based regression model for quality assessment of generated images, leveraging rich visual and textual knowledge encapsulated in CLIP. First, We design various prompts representing different quality levels to input into CLIP’s text encoder, mitigating semantic ambiguities. Second, by introducing a learnable prompts strategy and utilizing multiple quality-related auxiliary prompts, we make full use of CLIP’s textual knowledge. Last, our regression network then maps CLIP features to quality scores, effectively adapting CLIP’s capabilities to the task of generated image quality assessment.

<img src="https://github.com/wzczc/picgo-imgbed/blob/main/img/clip-agiqa.png" style="zoom: 50%;" />

<img src="https://github.com/wzczc/picgo-imgbed/blob/main/img/example.png" style="zoom:50%;" />

### Dependencies and Installation

This repo is based on [CoOp](https://github.com/KaiyangZhou/CoOp). Follow [this](https://github.com/KaiyangZhou/CoOp?tab=readme-ov-file#how-to-install) to install.

## Datasets

Download datasets [AGIQA-3k](https://github.com/lcysyzxdxc/AGIQA-3k-Database) and [AIGCIQA2023](https://github.com/wangjiarui153/AIGCIQA2023).

For AIGCIQA2023, use `mat2csv.py` to convert the MOS data in `.mat` format  to `.csv` format.

## Train and test

### AGIQA-3k

```shell
bash scripts/main.sh AGIQA vit_b16_ep100 end 16 1 False
```

<img src="https://github.com/wzczc/picgo-imgbed/blob/main/img/agiqa3k.png" style="zoom:50%;" />

### AIGCIQA2023

* For quality assessment：

  ```shell
  bash scripts/main.sh AIGCIQA2023 vit_b16_ep100 end 16 1 False
  ```

* For authenticity assessment：

  First modify lines 23 and 24 in `datasets/AIGCIQA2023.py` , then run：

  ```bash
  bash scripts/main.sh AIGCIQA2023 vit_b16_ep100 end 16 1 False
  ```

<img src="https://github.com/wzczc/picgo-imgbed/blob/main/img/aigciqa2023.png" style="zoom:50%;" />

For more details, please refer to [this](https://github.com/KaiyangZhou/CoOp/blob/main/COOP.md) .

## Acknowledgement

This project is based on [CoOp](https://github.com/KaiyangZhou/CoOp). Thanks for their awesome work.

## Citation

If our work is useful for your research, please consider citing：

```
@article{tang2024clip,
  title={CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP},
  author={Tang, Zhenchen and Wang, Zichuan and Peng, Bo and Dong, Jing},
  journal={arXiv preprint arXiv:2408.15098},
  year={2024}
}
```

## Contact

For any questions, feel free to contact: `wangzichuan2024@ia.ac.cn` .
