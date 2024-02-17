# %%
model_name = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo"  # Change to your preferred model

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import gc

class ModelModifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"":0})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.layer_snr = {}
        self.modified_layers = set()
        self.original_weights = {}

    def calculate_snr_for_layer(self, layer_type, layer_number):
        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                weights = module.weight.double()
                S = torch.linalg.svdvals(weights)
                weights = weights.detach().cpu()
                S = S.detach().cpu()
                sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)

                signal = S[S > mp_threshold].sum()
                noise = S[S <= mp_threshold].sum()
                snr = signal / noise if noise != 0 else float('inf')
                del S, weights
                torch.cuda.empty_cache()  # Clear PyTorch's CUDA memory cache
                gc.collect()
                return snr

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta))**2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75, dim=-1)
        q25 = torch.quantile(S, 0.25, dim=-1)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                print(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                weights = module.weight.double()
                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(sigma_estimated_full_iqr, n, m)

                S_reduced = S[S > mp_threshold_full_iqr]
                k = len(S_reduced)
                S[:k] = S_reduced
                S[k:] = 0
                print(f"Reduced from {S.shape} to {k}")

                reconstructed_weights = U @ torch.diag(S) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)
                return True

    def restore_model_original_layer(self, layer_type, layer_number):
        layer_name = f"{layer_type}_{layer_number}"
        if layer_name in self.original_weights:
            module = getattr(self.model, layer_name)
            module.weight = torch.nn.Parameter(self.original_weights[layer_name])
            print(f"Restored original weights for layer: {layer_name}")
            if layer_name in self.modified_layers:
                self.modified_layers.remove(layer_name)
        else:
            print(f"No original weights saved for layer: {layer_name}")

    def calculate_model_perplexity(self, datasets=['gsm8k'], seqlen=32, use_cuda_graph=False, use_flash_attn=False):
        model = self.model
        model_str = self.model_name
        acc_loss = 0.0
        total_samples = 0

        for dataset in datasets:
            input_tok = gptq_data_utils_math.get_test_tokens(dataset, seed=0, seqlen=seqlen, model=model_str)
            total_length = input_tok.size(0)
            nsamples = total_length // seqlen
            rest = total_length % seqlen

            if rest != 0:
                input_tok = input_tok[:-rest]

            input_tok = input_tok.view(-1, seqlen)
            total_samples += nsamples

            loss_fct = torch.nn.CrossEntropyLoss().cuda()
            progress = tqdm(range(nsamples))
            for ii in progress:
                input = input_tok[ii, :].cuda().view(1, -1)
                output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                acc_loss += loss.item()
                progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        return ppl

    def assess_layers_snr(self, layer_types, layer_numbers):
        for name, _ in self.model.named_modules():
            for layer_number in layer_numbers:
                for layer_type in layer_types:
                    if layer_type in name and str(layer_number) in name:
                        layer_name = f"{layer_type}+{layer_number}"
                        print("*" * 50, flush=True)
                        print(f"Calculating Signal to Noise Ratio at layer {layer_name}", flush=True)

                        # Additional RMT complexity: Calculate the average singular value
                        average_s_value = self.average_singular_value(S)
                        print(f"Average Singular Value at layer {layer_name}: {average_s_value}", flush=True)

                        # Additional RMT complexity: Calculate the Tracy-Widom distribution
                        tracy_widom_values = self.tracy_widom_distribution(beta=1, size=len(S))
                        print(f"Tracy-Widom Distribution Values at layer {layer_name}: {tracy_widom_values}", flush=True)

                        snr = self.calculate_snr_for_layer(layer_type, layer_number)
                        self.layer_snr[layer_name] = snr
                        print(f"Signal to Noise Ratio at layer {layer_name} = {snr}", flush=True)
                        print("*" * 50, flush=True)

    def select_layers_for_modification(self, k):
        sorted_layers = sorted(self.layer_snr.items(), key=lambda x: x[1], reverse=False)
        return [layer[0] for layer in sorted_layers[:k]]

    def test_and_modify_layers(self, candidate_layers):
        initial_perplexity = self.calculate_model_perplexity()
        print(f"Initial Model Perplexity: {initial_perplexity}")

        for layer in candidate_layers:
            layer_type = layer.split("+")[0]
            layer_number = layer.split("+")[1]
            self.update_model_reduce_layer(layer_type=layer_type, layer_number=layer_number)

            new_perplexity = self.calculate_model_perplexity()
            print(f"Tested Model Perplexity after modifying {layer}: {new_perplexity}")

            if new_perplexity > initial_perplexity:
                self.restore_model_original_layer(layer_type=layer_type, layer_number=layer_number)
                print(f"Reverted changes in {layer} due to lack of improvement.", flush=True)
            else:
                initial_perplexity = new_perplexity
                print(f"Modification kept for {layer}. New baseline perplexity: {initial_perplexity}", flush=True)

    def save_model(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    # Additional RMT complexity: Function to retrieve singular values for a specific layer
    def get_singular_values_for_layer(self, layer_type, layer_number):
        for name, module in self.model.named_modules():
            if layer_type in name and str(layer_number) in name:
                weights = module.weight.double()
                S = torch.linalg.svdvals(weights)
                return S

    # Additional RMT complexity: Advanced RMT analysis on a specific layer
    def advanced_rmt_analysis(self, layer_type, layer_number, singular_values):
        layer_name = f"{layer_type}+{layer_number}"
        print(f"Performing Advanced RMT Analysis on layer {layer_name}")

        # Additional RMT complexity: Calculate the average singular value
        average_s_value = self.average_singular_value(singular_values)
        print(f"Advanced RMT - Average Singular Value at layer {layer_name}: {average_s_value}", flush=True)

        # Additional RMT complexity: Calculate the Tracy-Widom distribution
        tracy_widom_values = self.tracy_widom_distribution(beta=1, size=len(singular_values))
        print(f"Advanced RMT - Tracy-Widom Distribution Values at layer {layer_name}: {tracy_widom_values}", flush=True)

# Usage
modifier = ModelModifier(model_name)

# %%
layer_numbers = list(range(31, -1, -1))
layer_numbers = [f".{l}." for l in layer_numbers]
print(layer_numbers)

layer_types = ['mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
               'self_attn.o_proj']

modifier.assess_layers_snr(layer_types, layer_numbers)
top_k_layers = modifier.select_layers_for_modification(15)  # Select top 15 layers
print(top_k_layers, flush=True)

modifier.test_and_modify_layers(top_k_layers)
# %%
modifier.save_model("laser_model")
