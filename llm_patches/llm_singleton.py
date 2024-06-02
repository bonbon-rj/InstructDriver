import socket
from llm_patches import misc
from accessory.util.tensor_type import default_tensor_type
from accessory.model.meta import MetaModel
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from fairscale.nn.model_parallel import initialize as fs_init
import torch

class LLaMA2Parameters:
    def __init__(self, llama_config, tokenizer_path, pretrained_path):

        # model parameters
        self.llama_type = 'llama_peft' # llama llama_peft
        self.llama_config = llama_config
        self.tokenizer_path = tokenizer_path
        self.pretrained_path = pretrained_path
        self.device = 'cuda'
        
        self.model_parallel_size = 1
        # self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = 'env://'
        self.dtype = "bf16" # ["fp16", "bf16"]
        self.quant = False

        self.max_seq_len = 12288
        self.max_gen_len = 3072
        
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def load_llm_and_args():

    port = get_free_port()
    
    
    llama_config=''
    lora_config=''
    tokenizer_path=''
    pretrained_path=''
    args = LLaMA2Parameters([llama_config, lora_config], tokenizer_path, [pretrained_path])
    args.available_dist_url = f'tcp://127.0.0.1:{port}'
    

    misc.init_distributed_mode_multi_url(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    args.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[args.dtype]
    with default_tensor_type(dtype=args.target_dtype, device="cpu" if args.quant else "cuda"):
        model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False, max_seq_len = args.max_seq_len)
    # print(f"load pretrained from {self.args.pretrained_path}")
    load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
    # print("load result: ", load_result)

    if args.quant:
        print("Quantizing model to 4bit!")
        from accessory.util.quant import quantize
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False, 
                "load_in_4bit": True, 
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)

    # print("Model = %s" % str(self.model))
    model.bfloat16().cuda()

    return model, args

class LLMSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMSingleton, cls).__new__(cls)

            cls._instance.model, cls._instance.args = load_llm_and_args()
        return cls._instance

    @staticmethod
    def get_llm_and_args():
        return LLMSingleton()._instance.model, LLMSingleton()._instance.args