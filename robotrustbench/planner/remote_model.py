import json
import requests
import sys
import os
import base64
import anthropic
import google.generativeai as genai
from openai import OpenAI
import typing_extensions as typing
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from robotrustbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from robotrustbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from robotrustbench.planner.planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, \
                                             ActionPlan_1_manip, ActionPlan_manip, ActionPlan_lang_manip, fix_json

temperature = 0
max_completion_tokens = 2048
remote_url = os.environ.get('remote_url')

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=2,
        task_type=None # used to distinguish between manipulation and other environments
    ):
        self.qianfan_api_models = ["deepseek-vl2", "ernie-4.5-turbo-vl", "qwen2.5-vl-32b-instruct", "qwen2.5-vl-7b-instruct",
                              "qianfan-composition", "internvl3-38b", "internvl3-14b",
                              "llama-4-maverick-17b-128e-instruct",
                              "llama-4-scout-17b-16e-instruct",
                              "internvl2.5-38b-mpo",
                              "glm-4.5v"]

        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type

        if self.model_type == 'local':
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        else:
            if self.model_name in self.qianfan_api_models:
                """璋冪敤鐧惧害鍗冨竼deepseek-vl2妯″瀷鎻忚堪鏈湴鍥剧墖鍐呭"""
                # API绔偣
                self.url = "https://qianfan.baidubce.com/v2/chat/completions"

                # 鏋勫缓璇锋眰澶?                
                self.headers = {
                    "Authorization": "",
                    "Content-Type": "application/json"
                }
                self.model = None
            elif "claude" in self.model_name or "gpt" in self.model_name or "gemini" in self.model_name or "qwen-vl-max" == self.model_name or "o3" in self.model_name or "o4" in self.model_name or "deepseek" in self.model_name:
                self.model = OpenAI(
                    base_url="https://4.0.wokaai.com/v1/",
                    api_key="", 
                    # api_key="",  # ModelScope Token
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        else:
            if self.model_name in self.qianfan_api_models:
                return self._call_qianfan(message_history)
            elif "claude" in self.model_name or "gpt" in self.model_name or "gemini" in self.model_name or "qwen-vl-max" == self.model_name or "o3" in self.model_name or "o4" in self.model_name or "deepseek" in self.model_name:
                return self._call_wokaai(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            #     return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            #     return self._call_intern38b(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")


    def _call_wokaai(self, message_history: list):
        # if self.model_name == "gemini-2.5-flash":
        #     self.model_name = "gemini-2.5-flash-lite-preview-06-17"
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content

        return out


    def _call_qianfan(self, message_history: list):

        # 鏋勫缓璇锋眰浣?        
        payload = {
            "model": self.model_name,
            "messages": message_history
        }

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(payload)
            )

            response.raise_for_status()
            output = response.json()
            return output["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"API璇锋眰澶辫触: {e}")
            # 鎵撳嵃鍝嶅簲鍐呭甯姪璋冭瘯
            if hasattr(e, 'response'):
                try:
                    print(f"閿欒璇︽儏: {e.response.json()}")
                except:
                    print(f"鍝嶅簲鍐呭: {e.response.text}")
            return None

    def _call_local(self, message_history: list):
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=temperature,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        return out

    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.task_type == 'manip':
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):

        # if not self.language_only:
        #     if self.task_type == 'manip':
        #         response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
        #     else:
        #         response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        # else:
        #     if self.task_type == 'manip':
        #         response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
        #     else:
        #         response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content

        return out
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                temperature = temperature
            )
            out = response.choices[0].message.content
            
        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                temperature = temperature
            )
            out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            temperature=temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out



if __name__ == "__main__":

    model = RemoteModel(
        'Qwen/Qwen2-VL-72B-Instruct', #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True #False
    )#'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("../../evaluator/midlevel/output.png")
        
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture?? {template}?"
            },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)


