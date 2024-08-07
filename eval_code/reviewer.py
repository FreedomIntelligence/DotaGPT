## llm
## Base
import os
from pathlib import Path
from openai import OpenAI
import json
from prompt import prompts
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from collections import defaultdict
from statistics import mean, stdev
import re
import argparse
import yaml
import logging
import datasets

def setup_logging(log_directory, debug_mode, log_filename="reviewer.log"):
    log_file_path = os.path.join(log_directory, log_filename)
    logger = logging.getLogger()
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

class SampleProcessor:
    def __init__(self, eval_set: str, data_dir: Path, result_dir: Path, system_prompt: str, eval_prompt: str, turn_type:str, turn_num: int):
        self.eval_set = eval_set
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.eval_prompt = eval_prompt
        self.system_prompt = system_prompt
        self.turn_type = turn_type
        self.turn_num = turn_num
        self.sample_list = []
        self.eval_list = []
        self.existing_samples = []
        self.processed_ids = set()
    
    def _read_samples(self, model_name: str, repeat_idx) -> List:
        self.eval_list = []
        self.sample_list = []
        self.existing_samples = []
        self.processed_ids = set()

        self.sample_list = self._load_jsonl(self.data_dir / f"{model_name}.jsonl")
        # self.sample_list = self._load_from_huggingface(self.eval_set)
        if os.path.exists(self.result_dir / f"eval_{repeat_idx}.jsonl"):
            self.existing_samples = self._load_jsonl(self.result_dir / f"eval_{repeat_idx}.jsonl")
            self.processed_ids = {sample.get("id") for sample in self.existing_samples}
        else:
            self.existing_samples = []
        
        self.sample_list = self._filter_samples(self.sample_list)
    
    def _load_from_huggingface(self, dataset_name: str) -> List:
        dataset = datasets.load_dataset(dataset_name)
        return [dict(sample) for sample in dataset['test']]  

    def _load_jsonl(self, file_path: Path) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def preprocess_samples(self, model_name: str, repeat_idx) -> List:
        self._read_samples(model_name, repeat_idx)
        eval_list = []
        for sample in self.sample_list:
            content = self._create_eval_object(sample, self.turn_type, self.turn_num)
            messages = [
                {'role': 'system',
                        'content': self.system_prompt
                },
                {
                    'role': 'user',
                    'content': content
                }
            ]
            eval_list.append({
                'id': sample['id'],
                'messages': messages,
                'category': sample.get('category', '') 
            })
        self.sample_list = eval_list
        return self.sample_list, self.existing_samples

    def _create_eval_object(self, sample: dict, turn_type: str, turn_num: int) -> str:
        fields = {'reference': sample.get('reference')}
        if turn_type == "single":
            assert 'input' in sample and 'output' in sample, "Sample must have 'input' and 'output' keys for single turn evaluation."
            fields.update({'input': sample.get('input', ''), 'output': sample.get('output', '')})
        else:
            for i in range(1, turn_num + 1):
                assert f'turn_{i}_question' in sample and f'turn_{i}_answer' in sample, f"Sample must have 'turn_{i}_question' and 'turn_{i}_answer' keys for {turn_num} turn evaluation."
                fields.update({
                    f'turn_{i}_question': sample.get(f'turn_{i}_question', ''),
                    f'turn_{i}_answer': sample.get(f'turn_{i}_answer', '')})
        return self.eval_prompt.format(**fields)

    def _filter_samples(self, samples: list) -> list:
        filtered_samples = []
        for sample in samples:
            if sample['id'] not in self.processed_ids:
                filtered_samples.append(sample)
                self.processed_ids.add(sample['id'])
        return filtered_samples
    
class LLMEvaluator:
    def __init__(self, config: dict, n_process: int=1):
        self.config = config
        self.n_process = n_process
        self.total_cost = 0
        self._init_llm_evaluator(config)
        self.kwargs = config.get("kwargs", {})

    def _init_llm_evaluator(self, config: dict):
        ##base_url key args like temperature..
        raise NotImplementedError

#todo processed_samples_count
    def request_eval(self, existing_samples_list, sample_list, output_dir, retries=0):
        eval_results = []
        incomplete_samples = []
        processed_samples_count = 0
        if self.n_process > 1:
            max_workers = min(self.n_process, len(sample_list))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sample = {executor.submit(self._request_eval, sample): sample for sample in sample_list}
                for future in as_completed(tqdm(as_completed(future_to_sample), total=len(sample_list), desc="Evaluating samples")):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        if result:
                            sample['eval_result'] = result
                            eval_results.append(sample)
                            processed_samples_count += 1
                        else:
                            incomplete_samples.append(sample)
                    except Exception as exc:
                        logging.error(f"Sample evaluation failed for {sample}: {exc}")
                        incomplete_samples.append(sample)

                    # if processed_samples_count % 2 == 0: 
                    #     self._save_result(existing_samples_list + eval_results, output_dir)
        else:
            for sample in tqdm(sample_list, desc="Evaluating samples"):
                assert sample.get('messages') is not None, "Sample messages is missing"
                result = self._request_eval(sample)
                if result:
                    sample['eval_result'] = result
                    eval_results.append(sample)
                    processed_samples_count += 1
                else:
                    incomplete_samples.append(sample)
                # if processed_samples_count % 2 == 0:
                #     self._save_result(existing_samples_list + eval_results, output_dir)
        self._save_result(existing_samples_list + eval_results, output_dir)
        ## if the complete, retry 3 times
        if incomplete_samples:
            if retries  < 3:
                retries  += 1
                logging.debug(f"Retry attempt {retries} for missing samples...")
                self.request_eval(existing_samples_list + eval_results, incomplete_samples, output_dir)
            else:
                raise Exception("Max retries reached, evaluation incomplete")
        else:
            logging.debug("All samples evaluated successfully.")


    def _save_result(self, result, output_dir):
        with open(output_dir, 'w', encoding = 'utf-8') as f:
            for sample in result:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    def _request_eval(self, sample):
        raise NotImplementedError

class GPTEvaluator(LLMEvaluator):
    def __init__(self, config: dict, n_process: int=1):
        super().__init__(config, n_process)
        self.cost_per_thousand_input_tokens = 0.001
        self.cost_per_thousand_output_tokens = 0.002

    def _init_llm_evaluator(self, config: dict):
        logging.info(f"Config of GPTEvaluator:{self.config}")
        self.client = OpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                timeout=config.get("timeout", 60))    

    def _validate_response(self, response):
        try:
            match = re.search(r'\[\[(\d+)\]\]', response)
            if match:
                return float(match.group(1))
            else:
                return None
        except Exception as e:
            logging.error(f"Error extracting score: {e}")
            return None

    def _request_eval(self, sample):
        response = self.client.chat.completions.create(
            model=self.config.get("gpt_version"),
            messages=sample.get("messages"),
            **self.kwargs,
        )
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.total_cost += self._calculate_cost(prompt_tokens, completion_tokens)
        if response.choices[0].message.content and self._validate_response(response.choices[0].message.content):
            return response.choices[0].message.content
        else:
            return None
    
    def _calculate_cost(self, prompt_tokens, completion_tokens):
        input_cost = prompt_tokens * self.cost_per_thousand_input_tokens / 1000
        output_cost = completion_tokens * self.cost_per_thousand_output_tokens / 1000

        # Total cost
        total_cost = input_cost + output_cost
        return total_cost

class Stats_Calculator:
    def __init__(self, 
                 results_dir: Path,
                 n_repeat: int,
                 ):
        self.results_dir = results_dir
        self.n_repeat = n_repeat
    
    def _extract_score(self, eval_result: str) -> float:
        try:
            match = re.search(r'\[\[(\d+)\]\]', eval_result)
            if match:
                return float(match.group(1))
            else:
                return float('-inf')
        except Exception as e:
            logging.error(f"Error extracting score: {e}")
            return float('-inf')

    ##compute
    def calculate_stats(self, results_list: List[List[Dict]]):
        #category_scores should be a list of dict

        category_scores = defaultdict(lambda: defaultdict(list))

        for index, result_set in enumerate(results_list):
            for result in result_set:
                category = result.get('category', "overall")
                score = self._extract_score(result.get('eval_result', ""))
                category_scores[category][index].append(score)

        category_mean_scores = {}
        category_std_scores = {}
        for category, eval_dict in category_scores.items():
            scores_per_eval = [mean(scores) for scores in eval_dict.values() if scores]
            category_mean_scores[category] = mean(scores_per_eval)
            category_std_scores[category] = stdev(scores_per_eval) if len(scores_per_eval) > 1 else 0
        
        all_scores = [score for eval_dict in category_scores.values() for scores in eval_dict.values() for score in scores]
        logging.debug(f"all_scores: {all_scores}")
        overall_mean = mean(all_scores)
        overall_std = stdev(all_scores) if len(all_scores) > 1 else 0

        self.statistics = {
            'category_mean_scores': category_mean_scores,
            'category_std_scores': category_std_scores,
            'overall_mean': overall_mean,
            'overall_std': overall_std
        }

        self._save_stats()
    def _save_stats(self):
        stats_file = self.results_dir / 'stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=4)


class LLMReviewer:
    def __init__(self,
                  model_name:str,
                  eval_set:str,
                  config: dict,
                  turn_type: str,
                  n_processes: int=50,
                  n_repeat: int=1,
                  turn_num: int=0,
                  ):
        self.model_name = model_name
        self.eval_set = eval_set
        self.config = config
        self.n_processes = n_processes
        self.turn_num = turn_num
        self.n_repeat = n_repeat
        self.turn_type = turn_type
        self.eval_result_list = []
        
        self.eval_prompt = ""
        self.system_prompt = ""

        self._init_eval_prompt()

        self.data_dir = Path('data') / eval_set

        if self.turn_type == "single":
            self.output_dir = Path("output") / self.eval_set / self.model_name
        else:
            turn_path = f"{self.eval_set}-turn-{self.turn_num}"
            self.output_dir = Path("output") / turn_path / self.model_name

        self.result_dir = self.output_dir / 'results'
        self.stat_dir = self.output_dir / 'stat'

        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.stat_dir.mkdir(parents=True, exist_ok=True)

        self.sample_processor = SampleProcessor(self.eval_set, self.data_dir, self.result_dir, self.system_prompt, self.eval_prompt, self.turn_type, self.turn_num)
        self.evaluator = GPTEvaluator(self.config, self.n_processes)
        self.stats_calculator = Stats_Calculator(self.stat_dir, self.n_repeat)

    def _init_eval_prompt(self):
        prompt_suffix = ""
        if self.turn_type == "multi":
            prompt_suffix = f"_turn_{self.turn_num}"
        logging.info(f"prompt key:{self.turn_type}{prompt_suffix}")
        logging.info(f"prompts:{prompts}")
        prompt = prompts.get(f"{self.turn_type}{prompt_suffix}")
        logging.info(f"Using prompt: {prompt}")
        self.eval_prompt = prompt.get(f"prompt_template")

        self.system_prompt = prompt.get("system_prompt")
        
        if not self.eval_prompt:
            raise ValueError(f"eval_prompt is invalid")
    
    def _load_jsonl(self, file_path: Path) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
## todo  load_jsonl       
    def merge_all_outputs(self) -> List[List[Dict]]:
        results_list = []
        for repeat_idx in range(self.n_repeat):
            output_path = self.result_dir / f"eval_{repeat_idx}.jsonl"
            results_list.append(self._load_jsonl(output_path))
        return results_list

    def review(self):
        for repeat_idx in range(self.n_repeat):
            logging.info(f"Starting repetition {repeat_idx}: Reading samples...")
            eval_list, existing_samples = self.sample_processor.preprocess_samples(self.model_name, repeat_idx)
            logging.info(f"Preprocessing completed: eval_list length = {len(eval_list)}, existing_samples length = {len(existing_samples)}")
            
            if len(eval_list):
                self.evaluator.request_eval(existing_samples, eval_list, self.result_dir / f"eval_{repeat_idx}.jsonl")
            logging.info(f"Evaluation request completed for repetition {repeat_idx}.")
        logging.info(f"Total cost for this evaluation: {self.evaluator.total_cost}")
        logging.info("Statistics gathering started.")
        results_list = self.merge_all_outputs()
        self.stats_calculator.calculate_stats(results_list)

def main():
    parser = argparse.ArgumentParser(description="LLM Reviewer")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--eval_set", type=str, required=True, choices=["DoctorFLAN", "DotaBench"], help="Name of the evaluation set")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--turn_type", type=str, required=True, choices=["single", "multi"], help="Type of turn: single or multi")
    parser.add_argument("--n_processes", type=int, default=1, help="Number of processes for parallel processing")
    parser.add_argument("--n_repeat", type=int, default=1, help="Number of repeatitions for each sample")
    parser.add_argument("--turn_num", type=int, default=0, help="Number of turns for multi-turn evaluation")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logging")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    log_dir = args.log_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, args.debug_mode)
    logging.info(f"args:{args}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    reviewer = LLMReviewer(args.model_name, args.eval_set, config, args.turn_type, args.n_processes, args.n_repeat, args.turn_num)
    reviewer.review()

if __name__ == "__main__":
    main()