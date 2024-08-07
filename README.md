## Get Started

```bash
git clone https://github.com/FreedomIntelligence/DotaGPT.git
pip install -r requirements.txt
```

## Evaluation

### Step 1: Download and Prepare Data

Download the datasets from Hugging Face:
- [FreedomIntelligence/DoctorFLAN](https://huggingface.co/datasets/FreedomIntelligence/DoctorFLAN)
- [FreedomIntelligence/DotaBench](https://huggingface.co/datasets/FreedomIntelligence/DotaBench)

### Step 2: Generate and Position the Data

**Data Format**

For `DotaBench`, the data is structured as follows. Each entry is a JSON object representing a series of interaction turns with a reference answer:

```json
{
  "id": 0,
  "turn_1_question": "example question 1",
  "turn_1_answer": "[model-generated answer for turn 1]",
  "turn_2_question": "example question 2",
  "turn_2_answer": "[model-generated answer for turn 2]",
  "turn_3_question": "example question 3",
  "turn_3_answer": "[model-generated answer for turn 3]",
  "reference": "example reference"
}
```
Complete the fields: `turn_1_answer`, `turn_2_answer`, `turn_3_answer`.

For `DoctorFLAN`, the data format is as follows, with each entry representing a single-turn interaction:

```json
{
  "id": 0,
  "input": "example input",
  "output": "[model-generated output]",
  "reference": "example reference answer"
}
```
Complete the field: `output`.

Store the generated model responses in the location: `data/{eval_set}/{model_name}.jsonl`. Ensure that all required fields are correctly filled.

### Step 3: Configuration

Prepare a YAML configuration file specifying model details, API keys, etc. Example (`configs/eval.yaml`):

```yaml
api_key: "your-openai-api-key"
base_url: "https://api.openai.com"
gpt_version: "gpt-4"
```

### Step 4: Run the Evaluation

Execute the evaluator with the script `script/run.sh`, modifying parameters as necessary. Example command:

```bash
python eval_code/reviewer.py \
    --config configs/eval.yaml \
    --model_name Baichuan-13B-Chat \
    --eval_set DotaBench \
    --turn_type multi \
    --n_processes 2 \
    --n_repeat 2 \
    --turn_num 2
```

**Parameter Explanation**

- `--config`: Path to the configuration file.
- `--model_name`: Name of the model being evaluated.
- `--eval_set`: Evaluation dataset being used. Choose either `DoctorFLAN` or `DotaBench`.
- `--turn_type`: Type of interaction (single or multi-turn).
- `--n_processes`: Number of processes for parallel processing.
- `--n_repeat`: Number of repetitions for each sample.
- `--turn_num`: Number of turns for multi-turn evaluations.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests on GitHub to help improve this project.

## Citation
The code in this repository is mostly developed for or derived from the paper below. 
```text
@article{xie2024llms,
  title={LLMs for Doctors: Leveraging Medical LLMs to Assist Doctors, Not Replace Them},
  author={Xie, Wenya and Xiao, Qingying and Zheng, Yu and Wang, Xidong and Chen, Junying and Ji, Ke and Gao, Anningzhe and Wan, Xiang and Jiang, Feng and Wang, Benyou},
  journal={arXiv preprint arXiv:2406.18034},
  year={2024}
}
```
## License

This project is licensed under the MIT License.

## Contact Us

For inquiries, please create an issue in this repository or email the authors: wenyaxie023@gmail.com
