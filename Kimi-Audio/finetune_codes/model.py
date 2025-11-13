import os
import argparse
from typing import Optional, List
import shutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from .modeling_kimia import MoonshotKimiaForCausalLM


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)
        # 在初始化时就将 speech_encoder 转换为 bf16，确保所有参数都是 bf16
        # 这样 Flash Attention 才能正常工作
        speech_encoder = self.whisper_model.speech_encoder
        # 将整个 speech_encoder 转换为 bf16
        speech_encoder = speech_encoder.to(torch.bfloat16)
        # 确保所有参数都是 bf16（包括 conv1d 的 bias）
        for name, param in speech_encoder.named_parameters():
            if param.dtype.is_floating_point:
                param.data = param.data.to(torch.bfloat16)
        # 确保所有 buffer 也是 bf16
        for name, buffer in speech_encoder.named_buffers():
            if buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(torch.bfloat16)

    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model
    
    @staticmethod
    def export_model(input_dir, output_dir):
        print("Loading model from {}".format(input_dir))
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # whisper_input_feature 是原始音频（numpy 数组）
        # 需要先转换为 mel spectrogram，然后通过 speech_encoder
        # 但为了避免 whisper_model.forward() 内部的 bf16 转换，我们直接处理
        
        # 处理输入格式
        if isinstance(whisper_input_feature, (list, tuple)) and len(whisper_input_feature) > 0:
            if isinstance(whisper_input_feature[0], np.ndarray):
                audio = whisper_input_feature[0]
            else:
                audio = whisper_input_feature[0].cpu().numpy() if torch.is_tensor(whisper_input_feature[0]) else whisper_input_feature[0]
        else:
            audio = whisper_input_feature[0].cpu().numpy() if torch.is_tensor(whisper_input_feature) else whisper_input_feature
        
        # 将音频转换为 mel spectrogram（使用 float32 计算，然后转换为 bf16）
        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import log_mel_spectrogram, pad_or_trim, N_SAMPLES
        audio_tensor = torch.from_numpy(audio).to(torch.float32)
        pad_audio = pad_or_trim(audio_tensor, length=N_SAMPLES)
        mel = log_mel_spectrogram(pad_audio, device=torch.cuda.current_device())  # shape: [80, 3000]
        
        # 将 mel 转换为 bf16，确保类型一致（speech_encoder 已经在 __init__ 中转换为 bf16）
        mel_input = mel.unsqueeze(0).to(torch.cuda.current_device()).to(torch.bfloat16)  # shape: [1, 80, 3000]
        
        # speech_encoder 已经在 __init__ 中转换为 bf16，直接使用
        speech_encoder = self.whisper_model.speech_encoder
        
        # 使用 autocast 确保所有计算都是 bf16（Flash Attention 需要）
        try:
            # 新版本 API - 使用 bf16 autocast
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                whisper_feats = speech_encoder(
                    mel_input,
                    return_dict=True,
                ).last_hidden_state
        except (AttributeError, TypeError):
            # 兼容旧版本
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                whisper_feats = speech_encoder(
                    mel_input,
                    return_dict=True,
                ).last_hidden_state
        
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )
        
        # whisper_feats 已经是 bf16，直接使用
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """为生成准备输入，PEFT 需要此方法"""
        # 如果有 past_key_values，只使用最后一个 token
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] if isinstance(past_key_values[0], tuple) else past_key_values[0].shape[2]
            # 如果输入长度超过 past_length，只保留新的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
                input_ids = input_ids[:, remove_prefix_length:]
            else:
                # 默认行为：只保留最后一个 token
                input_ids = input_ids[:, -1:]
        
        # 如果提供了 inputs_embeds，使用它而不是 input_ids
        model_inputs = {"input_ids": input_ids}
        
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        
        # 添加其他 kwargs
        model_inputs.update(kwargs)
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """重新排序缓存以支持 beam search，PEFT 需要此方法"""
        reordered_past = ()
        for layer_past in past_key_values:
            if isinstance(layer_past, tuple):
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            else:
                reordered_past += (layer_past.index_select(0, beam_idx.to(layer_past.device)),)
        return reordered_past
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":

        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})

        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(args.input_dir, args.output_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")