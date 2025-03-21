# MCP-MedSAM

Pytorch Implementation of the paper "[MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day](https://arxiv.org/abs/2412.05888)". This paper introduces a new variant of MedSAM, integrating a lightweight pre-trained tiny ViT, two novel prompts (modality prompt and content prompt), and a modality-based data sampling strategy. These enhancements enable the model to achieve strong performance without long training time and large GPU resource consumption. Right now, we just publish the inference code with model files, the weight file is [here] (https://drive.google.com/drive/folders/1NW4aSNhk-dtiK-dicTAUp0g0eR2fryNi?usp=sharing). The training code and more instructions will come after the decision.



## Citation

```bash
@misc{lyu2024mcpmedsampowerfullightweightmedical,
      title={MCP-MedSAM: A Powerful Lightweight Medical Segment Anything Model Trained with a Single GPU in Just One Day}, 
      author={Donghang Lyu and Ruochen Gao and Marius Staring},
      year={2024},
      eprint={2412.05888},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05888}, 
}
```
