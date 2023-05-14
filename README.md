# Source code and data for *The Magic of IF: Investigating Causal Reasoning Abilities in Large Language Models of Code* (Findings of ACL 2023) 

---

## Dependencies
 - Python>=3.7
 - openai
 - backoff

## Abductive reasoning
Code and data of the abductive reasoning task are in the `abductive/` folder. 
 - `call_codex.py`: prompt generation and prediction with Codex
 - `call_davinci.py`: prompt generation and prediction with Davinci
 - `test_cleanup_no_label.json`: input of the test data
 - `test_cleanup_ref.json`: labels of the test data
The data files are downloaded from https://github.com/XiangLi1999/Diffusion-LM/tree/main/datasets/ROCstory/anlg/anlg

## Counterfactual reasoning
Code and data of the counterfactual reasoning task are in the `counterfactual/` folder. 
 - `call_codex.py`: prompt generation and prediction with Codex
 - `call_davinci.py`: prompt generation and prediction with Davinci
 - `test_data_new.json`: test data from https://github.com/qkaren/Counterfactual-StoryRW

## Generation results of Codex
Generation results of Codex are in the `codex_output/` folder.

## Citation
Please cite our paper if this repository inspires your work.
```
TBA
```

## Contact
If you have any questions regarding the code, please create an issue or contact the owner of this repository.