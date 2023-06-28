import transformers.models as tfm

from .gpt2 import GPT2LMHeadModel
from .gptneo import GPTNeoForCausalLM

HF_MODEL_MAP = {
    tfm.gpt2.GPT2LMHeadModel: GPT2LMHeadModel,
    tfm.gpt_neo.GPTNeoForCausalLM: GPTNeoForCausalLM,
}
