# DotaGPT

## Introduction
This repository contains code and data for the DotaGPT paper, a large model that can serve as a doctor's assistant and a real-world doctor workflow task benchmark.

## Motivation
- **Enhancing Diagnosis & Treatment:** Primary care and specialist doctors may sometimes lack a comprehensive understanding of certain diseases, potentially leading to misdiagnoses and inappropriate treatments. Large language models can offer more holistic suggestions and explanations, aiding doctors in making accurate diagnoses.

- **Reducing Physicians' Workload:** LLM can alleviate the burden on doctors by minimizing repetitive tasks, allowing them to focus on more intricate patient care aspects.

- **Need for Real-world Medical Benchmarks:** There's a noticeable gap in the availability of benchmarks that align closely with the actual workflow of doctors, making it challenging to measure the real-world performance of various medical models.

Given these motivations, we introduce the DotaGPT model, paired with a benchmark designed specifically to gauge the capability of medical models in addressing genuine medical challenges.
## Data
### Data Collection and Processing
We have collected data divided into three datasets: one primarily for CDS (clinical decision support) tasks, one mainly for encyclopedia question-answer tasks, and one for other medical text tasks.
![image](https://github.com/FreedomIntelligence/DotaGPT/assets/55481158/3adf7329-c527-497e-88d8-0c9bed96c41e)

### CDS task definition
We have established 18 CDS (clinical decision support) tasks following a top-down approach based on the complete patient consultation process, from triage to diagnosis, treatment, and finally to discharge.
<img src="https://github.com/FreedomIntelligence/DotaGPT/assets/55481158/405f19c1-d453-48b7-8dfe-60cc1f02c4c8" alt="image" width="75%">

## Example
![image](https://github.com/FreedomIntelligence/DotaGPT/assets/55481158/130239da-9220-42c1-b227-61f922677b17)

Please note that the results showcased here are still in progress and not yet complete.


## Citation
Please cite the following if you found DotaGPT useful in your research.
```

```

## License


## Contact Us
To contact us feel free to create an Issue in this repository, or email the respective authors that contributed to this code base: 3046809534@qq.com
