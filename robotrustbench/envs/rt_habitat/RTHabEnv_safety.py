"""
Habitat Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the habitat framework. It supports various object interactions and task scenarios.
The code is based on https://github.com/facebookresearch/habitat-lab and https://github.com/apple/ml-llarp 

Dependencies:
- habitat-lab
- gym
- numpy
- PIL
"""
import gym
import os
import time
import json
import yaml
import imageio
from PIL import Image
import numpy as np
import habitat
import hydra
from habitat.datasets import make_dataset
from robotrustbench.envs.rt_habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
)
from habitat.gym.gym_definitions import _add_sim_sensor_to_config
from omegaconf import OmegaConf

from habitat_sim.utils import viz_utils as vut
from robotrustbench.envs.rt_habitat.config import default_structured_configs
import robotrustbench.envs.rt_habitat.predicate_task
import robotrustbench.envs.rt_habitat.config
import robotrustbench.envs.rt_habitat.measures
from robotrustbench.envs.rt_habitat.utils import observations_to_image, merge_to_file, draw_text
from robotrustbench.main import logger

import cv2

HABITAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/task/language_rearrangement.yaml')


ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction',
        'spatial_relationship', 'visual_appearance', 'long_horizon' , 'robust_word',
        'single_robust_test','robust_robust_error','robust_robust_redu','robust_robust_sema','robust_robust_raw','safety'
    ]
# 鐢熸垚 safety_1_1 鍒?safety_1_10 鍜?safety_2_1 鍒?safety_2_10
safety_entries = [f"safety_{i}_{j}" for i in [1, 2] for j in range(1, 11)]

# 娣诲姞鍒板師鍒楄〃
ValidEvalSets.extend(safety_entries)

def add_receptacle(string, skill):
    if 'table_0' in skill[1][0]:
        string += 'table ' + skill[1][0].split('table_0')[1]
    elif 'fridge' in skill[1][0]:
        string += 'refrigerator push point'
    elif 'refrigerator' in skill[1][0]:
        string += 'refrigerator'
    elif 'drawer_right' in skill[1][0]:
        string += 'right drawer of the kitchen counter'
    elif 'drawer_left' in skill[1][0]:
        string += 'left drawer of the kitchen counter'
    elif 'chair_0' in skill[1][0]:
        string += 'chair ' + skill[1][0].split('chair_0')[1]
    elif 'tvstand' in skill[1][0]:
        string += 'TV stand'
    elif 'counter_left' in skill[1][0]:
        string += 'left counter in the kitchen'
    elif 'counter_right' in skill[1][0]:
        string += 'right counter in the kitchen'
    elif 'sink' in skill[1][0]:
        string += 'sink in the kitchen'
    elif 'sofa' in skill[1][0]:
        string += 'sofa'
    elif 'cab' in skill[1][0]:
        string += 'cabinet ' + skill[1][0].split('_')[-1]
    else:
        raise NotImplementedError
    return string


def transform_action_to_natural_language(skill_set):
    language_skill_set = []
    for skill in skill_set:
        if 'nav' in skill[0]:
            string = 'navigate to the '
            string = add_receptacle(string, skill)
        elif 'pick' in skill[0]:
            string = 'pick up the ' + skill[0].split('_')[1]
        elif 'open' in skill[0]:
            string = 'open the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'close' in skill[0]:
            string = 'close the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'place' in skill[0]:
            string = 'place at the '
            string = add_receptacle(string, skill)
        else:
            raise NotImplementedError

        language_skill_set.append(string)
    return language_skill_set



class RTHabEnv(gym.Env):
    def __init__(self, eval_set='train', exp_name='', down_sample_ratio=1.0,
                 start_epi_index=0, resolution=500, recording=False,
                 perturbation_type='none', dynamic_perturbation=False,
                 perturbation_config_path=None,dataset_name="dataset.yaml",max_episode_steps=20):
        """
        Initialize the HabitatRearrange environment with dynamic perturbation support.

        Args:
            eval_set: Evaluation dataset name
            exp_name: Experiment name for logging
            down_sample_ratio: Ratio to downsample episodes
            start_epi_index: Starting episode index
            resolution: Image resolution
            recording: Whether to record video
            perturbation_type: Type of visual perturbation
            dynamic_perturbation: Whether to enable dynamic object perturbation
            perturbation_config_path: Path to dynamic perturbation configuration file
        """
        # load config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = habitat.get_config(HABITAT_CONFIG_PATH)
        _add_sim_sensor_to_config(self.config, ThirdRGBSensorConfig())
        # set the dataset
        assert eval_set in ValidEvalSets
        OmegaConf.set_readonly(self.config, False)
        self.config.habitat.dataset.data_path = os.path.join(os.path.dirname(__file__), 'datasets/{}.pickle'.format(eval_set))
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height = resolution
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width = resolution
        self.resolution = resolution

        OmegaConf.set_struct(self.config, False)
        self.config["habitat"]["dataset_name"] = dataset_name
        OmegaConf.set_struct(self.config, True)


        self.perturbation_type = perturbation_type
        if self.perturbation_type != 'none':
            logger.info(f"Visual perturbation '{self.perturbation_type}' is ACTIVE.")

        # Dynamic perturbation configuration
        self.dynamic_perturbation = dynamic_perturbation
        self.perturbation_config = {}
        self.perturbation_executed = False
        self.perturbation_sequence = []
        self.trigger_action = None

        if dynamic_perturbation:
            if perturbation_config_path is None:
                perturbation_config_path = os.path.join(os.path.dirname(__file__),
                                                        'config/dynamic_perturbation_config.yaml')
            self._load_perturbation_config(perturbation_config_path)
            logger.info(f"Dynamic perturbation is ACTIVE with config: {perturbation_config_path}")

        # modify config path to ease data loading
        self.dataset = make_dataset(self.config.habitat.dataset.type, config=self.config.habitat.dataset)

        # initilaize env
        self.env = habitat.gym.make_gym_from_config(self.config, self.dataset)
        self.observation_space = self.env.observation_space
        # action of LanguageRearangeEnv is discrete value from 0 to 69
        self.action_space = self.env.action_space

        # Episode tracking
        self.down_sample_ratio = down_sample_ratio
        self.number_of_episodes = self.env.number_of_episodes * down_sample_ratio
        self._reset = False
        self._current_episode_num = 0
        while start_epi_index >= 1 and self._current_episode_num < start_epi_index:
            self.env.reset(return_info=False)
            self._current_episode_num += 1

        self._current_step = 0
        self._max_episode_steps = max_episode_steps
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        # is holding an object
        self.is_holding = False
        self.episode_log = []

        # init instruction and skill sets
        self.episode_language_instruction = ''
        self.episode_data = None
        self.skill_set = self.env.env.env._env.task.actions['pddl_hl_action']._action_datas
        self.language_skill_set = transform_action_to_natural_language(self.skill_set)

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 1
        self.log_path = 'running/rt_habitat/{}'.format(exp_name)
        # video recorder
        self.recording = recording
        self.episode_video = []

    def _load_perturbation_config(self, config_path):
        """Load dynamic perturbation configuration file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.perturbation_config = config_data.get('perturbations', {})
            logger.info(f"Loaded perturbation config with {len(self.perturbation_config)} episodes")
        except Exception as e:
            logger.error(f"Failed to load perturbation config: {e}")
            self.perturbation_config = {}

    def _get_episode_perturbation(self):
        """Get perturbation configuration for current episode"""
        if not self.dynamic_perturbation:
            return None

        # Get current episode ID
        current_episode = self.current_episode()
        episode_id = current_episode.episode_id if hasattr(current_episode, 'episode_id') else str(self._current_episode_num)

        # Look up configuration
        if episode_id in self.perturbation_config:
            return self.perturbation_config[episode_id]

        # Also try matching by instruction pattern
        instruction = self.episode_language_instruction
        for ep_id, config in self.perturbation_config.items():
            if 'instruction_pattern' in config:
                if config['instruction_pattern'] in instruction:
                    return config

        return None

    # def _execute_perturbation_sequence(self):
    #     """Execute perturbation action sequence"""
    #     logger.info("Executing dynamic perturbation sequence...")
    #     perturbation_results = []

    #     # Save original state
    #     original_holding = self.is_holding
    #     original_step = self._current_step

    #     for action_info in self.perturbation_sequence:
    #         action_id = action_info['action_id']
    #         description = action_info.get('description', '')

    #         logger.info(f"  Perturbation action: {description} (id: {action_id})")

    #         # Execute environment action directly without incrementing main task steps
    #         obs, reward, done, info = self.env.step(action_id)

    #         # Update holding state
    #         if 'pick' in description:
    #             self.is_holding = True
    #         elif 'place' in description:
    #             self.is_holding = False

    #         perturbation_results.append({
    #             'action_id': action_id,
    #             'description': description,
    #             'success': not info.get('was_prev_action_invalid', False)
    #         })

    #     # Log perturbation execution
    #     self.episode_log.append({
    #         'perturbation_executed': True,
    #         'perturbation_actions': perturbation_results,
    #         'at_step': original_step
    #     })

    #     logger.info("Perturbation sequence completed")
    #     return perturbation_results

    def _execute_perturbation_sequence(self):
        """Execute perturbation action sequence and return the final observation."""
        logger.info("Executing dynamic perturbation sequence...")
        perturbation_results = []

        # We need a variable to hold the latest observation
        final_obs = None

        # Save original step to avoid counting perturbation steps
        original_step = self._current_step

        for action_info in self.perturbation_sequence:
            action_id = action_info['action_id']
            description = action_info.get('description', '')

            logger.info(f"  Perturbation action: {description} (id: {action_id})")

            # Execute environment action and CAPTURE the new observation
            obs, reward, done, info = self.env.step(action_id)
            final_obs = obs  # Update the final observation with the latest one

            # Update holding state based on the action description
            if 'pick' in description:
                self.is_holding = True
            elif 'place' in description:
                self.is_holding = False

            perturbation_results.append({
                'action_id': action_id,
                'description': description,
                'success': not info.get('was_prev_action_invalid', False)
            })

        # Log perturbation execution
        self.episode_log.append({
            'perturbation_executed': True,
            'perturbation_actions': perturbation_results,
            'at_step': original_step
        })

        logger.info("Perturbation sequence completed")
        # Return the observation from the VERY LAST perturbation action
        return final_obs

    def _apply_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Applies the configured visual perturbation to an image."""
        if self.perturbation_type == 'none':
            return image

        # Ensure image is in RGB format for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.perturbation_type == 'noise':
            # Add Gaussian noise
            mean = 0
            var = 100 # Moderate noise
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, image.shape).astype('uint8')
            noisy_image_bgr = cv2.add(image_bgr, gaussian)
            return cv2.cvtColor(noisy_image_bgr, cv2.COLOR_BGR2RGB)

        elif self.perturbation_type == 'blur':
            # Apply Gaussian blur
            blurred_image_bgr = cv2.GaussianBlur(image_bgr, (35, 35), 0)
            return cv2.cvtColor(blurred_image_bgr, cv2.COLOR_BGR2RGB)

        elif self.perturbation_type == 'occlusion':
            # Add a random black rectangle
            h, w, _ = image.shape
            occ_h = int(h * 0.35) # Occlude 35% of height
            occ_w = int(w * 0.35) # Occlude 35% of width
            x1 = np.random.randint(0, w - occ_w)
            y1 = np.random.randint(0, h - occ_h)
            occluded_image_bgr = image_bgr.copy()
            cv2.rectangle(occluded_image_bgr, (x1, y1), (x1 + occ_w, y1 + occ_h), (0, 0, 0), -1)
            return cv2.cvtColor(occluded_image_bgr, cv2.COLOR_BGR2RGB)

        elif self.perturbation_type == 'brigtness':
            hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * 0.1, 0, 255)
            darker_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            return cv2.cvtColor(darker_bgr, cv2.COLOR_BGR2RGB)

        elif self.perturbation_type == 'grayscale':
            # Convert to grayscale and then back to 3 channels for model compatibility
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray_3_channel_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(gray_3_channel_bgr, cv2.COLOR_BGR2RGB)

        elif self.perturbation_type == 'low_res':
            # Downsample and then upsample to create a pixelated effect
            h, w, _ = image.shape
            low_res_bgr = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
            upsampled_bgr = cv2.resize(low_res_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            return cv2.cvtColor(upsampled_bgr, cv2.COLOR_BGR2RGB)

        return image

    def current_episode(self, all_info: bool = False):
        return self.env.current_episode(all_info)

    def reset(self, **kwargs):
        """
        Reset the environment for a new episode. The env will iterate over all the task data from the dataset
        Returns: observation
        """
        assert self._current_episode_num <= self.number_of_episodes
        obs, info = self.env.reset(return_info=True, **kwargs)
        logger.info('Episode {}: {}'.format(str(self._current_episode_num), str(self.current_episode())))
        self.episode_language_instruction = info['lang_goal']
        self.episode_data = self.dataset.episodes[self._current_episode_num]
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        self.is_holding = False
        self._reset = True
        self.episode_log = []
        if self.recording:
            self.episode_video = []
        self._episode_start_time = time.time()

        # Reset perturbation state
        self.perturbation_executed = False
        self.perturbation_sequence = []
        self.trigger_action = None

        # Get perturbation configuration for current episode
        if self.dynamic_perturbation:
            perturbation_config = self._get_episode_perturbation()
            if perturbation_config:
                self.trigger_action = perturbation_config.get('trigger', {})
                self.perturbation_sequence = perturbation_config.get('perturbation_sequence', [])
                logger.info(f"Episode {self._current_episode_num - 1} has perturbation configured")
                logger.info(f"  Trigger: {self.trigger_action}")
                logger.info(f"  Sequence length: {len(self.perturbation_sequence)}")

        return obs

    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        if info['was_prev_action_invalid']:
            env_feedback = 'Last action is invalid.'
            if 'pick' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot pick any object when holding something. Please place the object before picking something.'
                else:
                    env_feedback += ' Robot cannot pick any object that is not near the robot. Navigate to other place to find the object.'
            elif 'place' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot place any object that is not near the robot. Navigate to other place to find the object.'
                else:
                    env_feedback += ' Robot cannot place any object when not holding something. Please pick the object before place it.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already open or the robot is not near the receptacle."
            elif 'close' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already closed or the robot is not near the receptacle."
        else:
            env_feedback = 'Last action executed successfully'
            if 'pick' in info['action'] and self.feedback_verbosity:
                self.is_holding = True
                env_feedback += ' and you are holding {}.'.format(info['action'].split('(')[0].split('_')[1])
            elif 'place' in info['action'] and self.feedback_verbosity:
                self.is_holding = False
                env_feedback += ' and you are holding nothing.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is open.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is open.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            elif 'close' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is closed.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is closed.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            else:
                env_feedback += '.'

        return env_feedback

    # def step(self, action, reasoning='', **kwargs):
    #     """
    #     Execute a single environment step with dynamic perturbation support.
    #     Args:
    #         action (int): Index of action in action space
    #         reasoning (str): Reasoning for the action
    #     Returns:
    #         tuple: (observation, reward, done, environment feedback)
    #     """
    #     assert self._reset, 'Reset env before stepping'

    #     # Check if we should execute perturbation
    #     should_perturb = False
    #     if (self.dynamic_perturbation and
    #         not self.perturbation_executed and
    #         self.trigger_action and
    #         self.perturbation_sequence):

    #         # Check if current action is the trigger action
    #         if action == self.trigger_action.get('action_id'):
    #             should_perturb = True
    #             logger.info(f"Trigger action detected: {action} - {self.language_skill_set[action]}")

    #     # Execute normal action
    #     self._current_step += 1
    #     obs, reward, done, info = self.env.step(action, **kwargs)

    #     if self.recording:
    #         self.episode_video.append(self.env.render("rgb_array"))

    #     if info['was_prev_action_invalid']:
    #         self._cur_invalid_actions += 1
    #     else:
    #         # If action succeeded and perturbation is needed, execute perturbation sequence
    #         if should_perturb:
    #             logger.info("Triggering dynamic perturbation after successful action")
    #             self._execute_perturbation_sequence()
    #             self.perturbation_executed = True
    #             # Get observation after perturbation
    #             obs = self.env._env.env._env.task.observations[self.env._env.env._env._current_episode_step]

    #     # if exceed the max step
    #     if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
    #         done = True

    #     # env feedback
    #     env_feedback = self.get_env_feedback(info)

    #     # Add perturbation note to feedback if just executed
    #     if self.perturbation_executed and should_perturb:
    #         env_feedback += " [Note: Object positions may have changed]"

    #     info['env_feedback'] = env_feedback
    #     info['env_step'] = self._current_step
    #     info['episode_elapsed_seconds'] = time.time() - self._episode_start_time,
    #     info['action_id'] = action
    #     info['action_description'] = self.language_skill_set[action]
    #     info['reasoning'] = reasoning
    #     info['instruction'] = self.episode_language_instruction
    #     info['last_action_success'] = 1 - float(info['was_prev_action_invalid'])
    #     info['task_success'] = info['predicate_task_success']
    #     info['dynamic_perturbation_applied'] = self.perturbation_executed

    #     if info['task_success']:
    #         info['task_progress'] = 1.0

    #     self.episode_log.append(info)
    #     return obs, reward, done, info

    # def step(self, action, reasoning='', **kwargs):
    #     """
    #     Execute a single environment step with dynamic perturbation support.
    #     Args:
    #         action (int): Index of action in action space
    #         reasoning (str): Reasoning for the action
    #     Returns:
    #         tuple: (observation, reward, done, environment feedback)
    #     """
    #     assert self._reset, 'Reset env before stepping'

    #     # 1. 妫€鏌ュ綋鍓嶅姩浣滄槸鍚︽槸瑙﹀彂鍔ㄤ綔 (鍙鏌ユ剰鍥撅紝涓嶅叧蹇冨悗缁槸鍚︽垚鍔?
    #     # <--- KEY CHANGE: Logic to detect the trigger is now simpler --->
    #     should_perturb = False
    #     if (self.dynamic_perturbation and
    #         not self.perturbation_executed and
    #         self.trigger_action and
    #         self.perturbation_sequence and
    #         action == self.trigger_action.get('action_id')):

    #         should_perturb = True
    #         logger.info(f"Trigger action {action} ({self.language_skill_set[action]}) detected. Perturbation will be applied after this action.")

    #     # 2. 姝ｅ父鎵ц鏅鸿兘浣撻€夋嫨鐨勫姩浣?    #     self._current_step += 1
    #     obs, reward, done, info = self.env.step(action, **kwargs)

    #     if self.recording:
    #         self.episode_video.append(self.env.render("rgb_array"))

    #     # 3. 濡傛灉瑙﹀彂鏉′欢琚弧瓒筹紝鍒欏湪鏅鸿兘浣撳姩浣滀箣鍚庣珛鍗虫墽琛屾壈鍔?    #     # <--- KEY CHANGE: This block is now outside the success/failure check --->
    #     if should_perturb:
    #         logger.info("Applying dynamic perturbation sequence...")
    #         self._execute_perturbation_sequence()
    #         self.perturbation_executed = True
    #         # 鎵板姩鍚庯紝蹇呴』鑾峰彇鏈€鏂扮殑瑙傛祴锛屽惁鍒欐櫤鑳戒綋鐪嬪埌鐨勬槸鏃х殑鐢婚潰
    #         obs = self.env._env.env._env.task.observations[self.env._env.env._env._current_episode_step]
    #         # 鏄庣‘鍦板湪info涓爣璁版壈鍔ㄥ凡鍙戠敓
    #         info['dynamic_perturbation_applied'] = True

    #     # 4. 澶勭悊鏅鸿兘浣撳師濮嬪姩浣滅殑缁撴灉锛堟棤璁烘槸鍚﹀彂鐢熶簡鎵板姩锛?    #     if info['was_prev_action_invalid']:
    #         self._cur_invalid_actions += 1

    #     # 妫€鏌ユ槸鍚﹀洜涓烘鏁拌秴闄愭垨鏃犳晥鍔ㄤ綔杩囧鑰岀粓姝?    #     if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
    #         done = True

    #     # 5. 鐢熸垚鐜鍙嶉
    #     env_feedback = self.get_env_feedback(info)

    #     # 濡傛灉鍒氬垰鎵ц浜嗘壈鍔紝鍦ㄥ弽棣堜腑娣诲姞鎻愮ず
    #     if self.perturbation_executed and should_perturb:
    #         env_feedback += " [Note: The environment has changed unexpectedly.]"

    #     # 6. 鏁寸悊骞惰繑鍥炴渶缁堜俊鎭?    #     info['env_feedback'] = env_feedback
    #     info['env_step'] = self._current_step
    #     info['episode_elapsed_seconds'] = time.time() - self._episode_start_time,
    #     info['action_id'] = action
    #     info['action_description'] = self.language_skill_set[action]
    #     info['reasoning'] = reasoning
    #     info['instruction'] = self.episode_language_instruction
    #     info['last_action_success'] = 1 - float(info['was_prev_action_invalid'])
    #     info['task_success'] = info['predicate_task_success']
    #     # 纭繚杩欎釜鏍囧織鍦ㄦ壈鍔ㄥ彂鐢熸椂琚纭缃?    #     info['dynamic_perturbation_applied'] = self.perturbation_executed and should_perturb

    #     if info['task_success']:
    #         info['task_progress'] = 1.0

    #     self.episode_log.append(info)
    #     return obs, reward, done, info

    def step(self, action, reasoning='', **kwargs):
        """
        Execute a single environment step with dynamic perturbation support.
        Args:
            action (int): Index of action in action space
            reasoning (str): Reasoning for the action
        Returns:
            tuple: (observation, reward, done, environment feedback)
        """
        assert self._reset, 'Reset env before stepping'

        should_perturb = False
        if (self.dynamic_perturbation and
            not self.perturbation_executed and
            self.trigger_action and
            self.perturbation_sequence and
            action == self.trigger_action.get('action_id')):

            should_perturb = True
            logger.info(f"Trigger action {action} ({self.language_skill_set[action]}) detected. Perturbation will be applied after this action.")

        self._current_step += 1
        obs, reward, done, info = self.env.step(action, **kwargs)

        if self.recording:
            self.episode_video.append(self.env.render("rgb_array"))

        if should_perturb:
            logger.info("Applying dynamic perturbation sequence...")
            # <--- KEY CHANGE: Get the new observation directly from the sequence function --->
            new_obs_after_perturbation = self._execute_perturbation_sequence()
            if new_obs_after_perturbation is not None:
                obs = new_obs_after_perturbation # Overwrite the old observation with the new one

            self.perturbation_executed = True
            info['dynamic_perturbation_applied'] = True

        if info['was_prev_action_invalid']:
            self._cur_invalid_actions += 1

        if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
            done = True

        env_feedback = self.get_env_feedback(info)

        if self.perturbation_executed and should_perturb:
            env_feedback += " [Note: The environment has changed unexpectedly.]"

        info['env_feedback'] = env_feedback
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time,
        info['action_id'] = action
        info['action_description'] = self.language_skill_set[action]
        info['reasoning'] = reasoning
        info['instruction'] = self.episode_language_instruction
        info['last_action_success'] = 1 - float(info['was_prev_action_invalid'])
        info['task_success'] = info['predicate_task_success']
        info['dynamic_perturbation_applied'] = self.perturbation_executed and should_perturb

        if info['task_success']:
            info['task_progress'] = 1.0

        self.episode_log.append(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_image(self, obs, key='head_rgb'):
        """Save current agent observation as a PNG image."""
        folder = self.log_path + '/images/episode_{}'.format(self._current_episode_num)
        if not os.path.exists(folder):
            os.makedirs(folder)

        original_image_array = observations_to_image(obs, key)
        perturbed_image_array = self._apply_perturbation(original_image_array)
        img = Image.fromarray(perturbed_image_array)

        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(self._current_episode_num, self._current_step))
        img.save(image_path)
        return image_path

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Add perturbation summary information
        if self.dynamic_perturbation:
            perturbation_summary = {
                'dynamic_perturbation_enabled': True,
                'perturbation_executed': self.perturbation_executed,
                'trigger_action': self.trigger_action,
                'perturbation_sequence_length': len(self.perturbation_sequence)
            }
            self.episode_log.insert(0, perturbation_summary)

        filename = 'episode_{}_step_{}.json'.format(self._current_episode_num, self._current_step)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'w', encoding='utf-8') as f:
                for item in self.episode_log:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        if len(self.episode_video):
            folder = self.log_path + '/video'
            if not os.path.exists(folder):
                os.makedirs(folder)
            video_writer = imageio.get_writer(os.path.join(folder, 'video_episode_{}_steps_{}.mp4'.format(self._current_episode_num, self._current_step)), fps=30)
            for data in self.episode_video:
                video_writer.append_data(data)
            video_writer.close()

    def render(self, mode: str = "rgb"):
        return self.env.render(mode)

    def close(self) -> None:
        """Terminate the environment."""
        self.env.close()


if __name__ == '__main__':
    """
    Example usage of the RTHabEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = RTHabEnv(eval_set='base', dynamic_perturbation=True)
    obs = env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        env.save_image(obs)
        action = int(input('action id: '))
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break

        obs_new, reward, done, info = env.step(action)
        print(reward, done, info)
        env.save_image(obs_new)
        if done:
            break
    env.close()




