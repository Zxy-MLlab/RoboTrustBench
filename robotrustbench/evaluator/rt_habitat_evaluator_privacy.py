import os
import numpy as np
from tqdm import tqdm
import time
import json
from robotrustbench.envs.rt_habitat.RTHabEnv_privacy import RTHabEnv, ValidEvalSets
from robotrustbench.planner.vlm_planner import VLMPlanner
from robotrustbench.evaluator.summarize_result import average_json_values
from robotrustbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from robotrustbench.evaluator.config.system_prompts import trust_habitat_single_action_system_prompt, trust_habitat_multi_system_prompt, trust_habitat_single_action_with_constraints_system_prompt, trust_habitat_multi_with_constraints_system_prompt
from robotrustbench.main import logger

link_path = os.path.join(os.path.dirname(__file__), '../envs/rt_habitat/data')
try:
    os.symlink(link_path, 'data')
except FileExistsError:
    pass 


example_path = os.path.join(os.path.dirname(__file__), 'config/habitat_examples.json')
examples = json.load(open(example_path, 'r+'))



class EB_HabitatEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None
        if self.config["trust"]:
            self.system_prompt = trust_habitat_single_action_with_constraints_system_prompt if self.config[
                                                                                  "action_step"] == "single" else trust_habitat_multi_with_constraints_system_prompt

        else:
            self.system_prompt = trust_habitat_single_action_system_prompt if self.config["action_step"] == "single" else trust_habitat_multi_system_prompt

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
        
    def save_episode_metric(self, episode_info):
        filename = 'episode_{}_final_res.json'.format(self.env._current_episode_num)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        # If user provided a single string via CLI (e.g. eval_sets=privacy),
        # treat it as a single-element list instead of iterating into chars.
        if isinstance(valid_eval_sets, str):
            valid_eval_sets = [valid_eval_sets]
        valid_eval_sets = list(valid_eval_sets)
        if len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets
            
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = RTHabEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], exp_name=exp_name,
                                             start_epi_index=self.config.get('start_epi_index', 0), resolution=self.config.get('resolution', 500), dataset_name=self.config["dataset_name"], max_episode_steps=self.config.max_episode_steps)

            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, self.system_prompt, examples, n_shot=self.config['n_shots'], obs_key='head_rgb',
                                                 chat_history=self.config['chat_history'], language_only=self.config['language_only'], 
                                                 use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), tp=self.config.get('tp', 1))

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            done = False
            while not done:
                try: 
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")

                    if action == -2: # empty plan stop here
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'subgoal_reward': episode_info.get("subgoal_reward", 0),
                            'env_step': self.env._current_step,
                        }
                        break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'subgoal_reward': episode_info.get("subgoal_reward", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    # multiple actions
                    if type(action) == list:
                        if any(x not in range(0, len(self.env.language_skill_set)) for x in action):
                            self.env._cur_invalid_actions += 1
                            episode_info['reward'].append(-1)
                            episode_info['num_invalid_actions'] += 1
                            self.env.episode_log.append({
                                'last_action_success': 0.0,
                                'action_id': -1,
                                'action_description': 'invalid action',
                                'reasoning': reasoning,
                            })
                            info = {
                                'task_success': episode_info.get('task_success', 0),
                                'task_progress': episode_info.get("task_progress", 0),
                                'subgoal_reward': episode_info.get("subgoal_reward", 0),
                                'env_step': self.env._current_step,
                            }
                            if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                                break
                            continue
                        for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            if done or info['last_action_success'] == 0:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                            
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    print(e)
                    time.sleep(30)

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['subgoal_reward'] = info['subgoal_reward']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)
            
            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--start_epi_index', type=int, help='Starting episode index.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='number of tensor parallel splits of the model parameters')
        parser.add_argument("--dataset_name", type=str, help='datasets.yaml')
        parser.add_argument("--action_step", type=str, help="single / multi")
        parser.add_argument("--max_episode_steps", type=int, help="max episode steps")
        return parser.parse_args()

    config = {
        'model_name': 'gpt-4o-mini',  # 'Qwen/Qwen2-VL-7B-Instruct', 
        'n_shots': 10,
        'down_sample_ratio': 1.0, 
        'model_type': 'remote', # 'local'
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0,  
        'start_epi_index': 0,
        'eval_sets': ['privacy2-1', 'privacy2-2', 'privacy2-3', 'privacy2-4', 'privacy2-5', 
                      'privacy2-6', 'privacy2-7', 'privacy2-8', 'privacy2-9', 'privacy2-10',
                      'privacy3-5-1', 'privacy3-5-2', 
                      'privacy3-4-1', 'privacy3-4-2', 
                      'privacy3-3-2', 'privacy3-3-1', 
                      'privacy3-2-2', 'privacy3-2-1', 
                      'privacy3-1-2', 'privacy3-1-1', 
                      'privacy3-6-1', 'privacy3-6-2', 
                      'privacy3-7-1', 'privacy3-7-2', 
                      'privacy3-8-1', 'privacy3-8-2', 
                      'privacy3-9-1', 'privacy3-9-2', 
                      'privacy3-10-1', 'privacy3-10-2', 
                      'privacy', 'base', 'common_sense', 'complex_instruction', 'spatial_relationship',  'visual_appearance' , 'long_horizon' ,
                      'privacy1' ,'privacy2', 'privacy3', 'privacy4', 'privacy5', 'privacy6', 'privacy7', 'privacy8', 'privacy9' , 'privacy10',
                      'test_safety2'
                      ],
        'multistep':0, 
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 1,
        "dataset_name": "dataset.yaml",
        "action_step": "single",
        "max_episode_steps": 20
    }
    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_HabitatEvaluator(config)
    evaluator.evaluate_main()

    try:
        os.unlink('data')
        print(f"The symbolic link {link_path} has been successfully removed.")
    except FileNotFoundError:
        print(f"Error: The symbolic link {link_path} does not exist.")
    except OSError as e:
        print(f"Error: {e}")








