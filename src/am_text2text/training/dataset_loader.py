import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import Dataset


def render_user_prompt(prompt_template: str, input_text: str, title_text: str) -> str:
    return prompt_template.replace("[input]", input_text).replace("[Title]", title_text)


def build_plain_causal_prompt(system_prompt: Optional[str], user_prompt: str) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"System:\n{system_prompt}")
    parts.append(f"User:\n{user_prompt}")
    parts.append("Assistant:\n")
    return "\n\n".join(parts)


class Seq2SeqAndCausalDataset(Dataset):
    def __init__(
        self,
        data_list: list[dict],
        tokenizer,
        max_input_length: int,
        max_output_length: int,
        model_type: str = "seq2seq",
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_inference: bool = False,
        use_chat_template: bool = True,
    ) -> None:
        if model_type not in {"seq2seq", "causal"}:
            raise ValueError("model_type must be 'seq2seq' or 'causal'")

        self.inputs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.attention_masks: list[Optional[torch.Tensor]] = []
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template

        for item in data_list:
            input_text = item.get("input_text", "")
            output_text = item.get("output_text", "")
            title_text = item.get("title_text", "")

            if self.model_type == "seq2seq":
                input_enc = tokenizer(
                    input_text,
                    max_length=max_input_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                label_enc = tokenizer(
                    output_text,
                    max_length=max_output_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                labels = label_enc["input_ids"][0].clone()
                labels[label_enc["attention_mask"][0] == 0] = -100
                self.inputs.append(input_enc["input_ids"][0])
                self.labels.append(labels)
                self.attention_masks.append(input_enc["attention_mask"][0])
                continue

            if prompt_template is None:
                raise ValueError("prompt_template is required for causal models")

            user_prompt = render_user_prompt(prompt_template, input_text, title_text)
            if self.use_chat_template:
                prompt_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                combined_text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": output_text},
                    ],
                    tokenize=False,
                )
            else:
                prompt_text = build_plain_causal_prompt(system_prompt, user_prompt)
                combined_text = prompt_text + output_text
                if self.tokenizer.eos_token:
                    combined_text += self.tokenizer.eos_token

            if is_inference:
                enc = tokenizer(
                    prompt_text,
                    max_length=max_input_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self._warn_max_sequence_length(max_input_length, prompt_text, "prompt")
                input_ids = enc["input_ids"][0]
                self.inputs.append(input_ids)
                self.labels.append(torch.full_like(input_ids, -100))
                self.attention_masks.append(enc["attention_mask"][0])
                continue

            enc = tokenizer(
                combined_text,
                max_length=max_input_length + max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self._warn_max_sequence_length(max_input_length + max_output_length, combined_text, "combined")

            input_ids = enc["input_ids"][0]
            labels = input_ids.clone()
            prefix_enc = tokenizer(
                prompt_text,
                max_length=max_input_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prefix_len = prefix_enc["input_ids"].shape[1]
            labels[:prefix_len] = -100

            self.inputs.append(input_ids)
            self.labels.append(labels)
            self.attention_masks.append(enc["attention_mask"][0])

    def _warn_max_sequence_length(self, max_len: int, text: str, name: str) -> None:
        needed = len(self.tokenizer.tokenize(text))
        if needed > max_len:
            logging.warning("[%s] length %s exceeds max_length %s", name, needed, max_len)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class Seq2SeqAndCausalDatasetLoader:
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer,
        max_input_length: int,
        max_output_length: int,
        prompt_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else None

    def load_dataset(
        self,
        dataset_name: str,
        split: str,
        model_type: str = "seq2seq",
        prompt_name: Optional[str] = None,
        system_prompt_name: Optional[str] = None,
        is_inference: Optional[bool] = None,
        use_chat_template: bool = True,
    ) -> Seq2SeqAndCausalDataset:
        dataset_path = self.data_dir / dataset_name / f"{dataset_name}_{split}.json"
        data_list = json.loads(dataset_path.read_text(encoding="utf-8"))

        prompt_template = None
        system_prompt = None
        if model_type == "causal":
            if self.prompt_dir is None:
                raise ValueError("prompt_dir is required for causal models")
            if prompt_name is None:
                raise ValueError("prompt_name is required for causal models")
            prompt_path = self.prompt_dir / f"{prompt_name}.txt"
            prompt_template = prompt_path.read_text(encoding="utf-8").strip()
            if system_prompt_name is not None:
                system_prompt_path = self.prompt_dir / f"{system_prompt_name}.txt"
                system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

        if is_inference is None:
            is_inference = split == "test"

        return Seq2SeqAndCausalDataset(
            data_list=data_list,
            tokenizer=self.tokenizer,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            model_type=model_type,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            is_inference=is_inference,
            use_chat_template=use_chat_template,
        )
