import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

from mteb import MTEB


class ModelModifier:
    def __init__(self, model_name, task, eval_split, metric):
        self.model_name = model_name
        print(model_name)

        self.model = SentenceTransformer(model_name)
        self.original_weights = {}
        self.modified_layers = set()
        self.failed_attempts = set()
        self.task = task
        self.eval_split = eval_split
        self.metric = metric

    def update_model_reduce_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        if layer_id in self.modified_layers:
            print(f"Layer {layer_id} has already been modified. Skipping.")
            return False

        for name, module in self.model[0].named_modules():
            if layer_type in name and str(layer_number) in name:
                print(f"Reconstructing layer: {name}")
                original_dtype = module.weight.dtype
                self.original_weights[name] = module.weight.detach().clone()
                weights = module.weight.double()
                U, S, V = torch.linalg.svd(weights, full_matrices=False)

                # Estimate sigma using the full IQR method
                sigma_estimated_full_iqr = self.estimate_sigma_with_full_iqr(S)

                # Calculate Marchenko-Pastur threshold
                n, m = weights.shape
                mp_threshold_full_iqr = self.marchenko_pastur_threshold(
                    sigma_estimated_full_iqr, n, m
                )

                # Retain only the singular values above the MP threshold
                S_reduced = torch.zeros_like(S)
                k = (S > mp_threshold_full_iqr).sum().item()
                S_reduced[:k] = S[:k]
                print(f"Reduced from {S.shape} to {k}")

                # Reconstruct the matrix using the thresholded singular values
                reconstructed_weights = U @ torch.diag(S_reduced) @ V
                reconstructed_weights = reconstructed_weights.to(original_dtype)
                module.weight = torch.nn.Parameter(reconstructed_weights)
                self.modified_layers.add(layer_id)
                return True

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    ## Calculate an estimate of the standard deviation of the singular values based on Inter Quantile Range

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = (
            iqr / 1.349
        )  ## 0.6745 * sigma is the expected range between the quantiles (Q1 and Q3)
        return sigma_estimated

    def restore_model_original_layer(self, layer_type, layer_number):
        layer_id = f"{layer_type}_{layer_number}"
        for name, module in self.model[0].named_modules():
            if layer_type in name and layer_number in name:
                if name in self.original_weights:
                    module.weight = torch.nn.Parameter(self.original_weights[name])
                    print(f"Restored original weights for layer: {name}")
                    if layer_id in self.modified_layers:
                        self.modified_layers.remove(layer_id)
                else:
                    print(f"No original weights saved for layer: {name}")

    def calculate_model_peformance(self):
        evaluation = MTEB(tasks=[self.task], task_langs=["en"])
        results = evaluation.run(
            self.model, eval_splits=[self.eval_split], overwrite_results=True
        )
        if self.metric not in results[self.task][self.eval_split]:
            return results[self.task][self.eval_split]["en"][self.metric]
        return results[self.task][self.eval_split][self.metric]

    ### Implement a Backward Search
    # Search for the optimal lower ranking approximations from the top layers downwards
    # Also, we try doing a greedy approach, in order to maximize the rank reduction.
    # We tune the compression rate based on Marchenko-Pastur Random Matrix Theory
    ######################################################################################

    def search_optimal_layer_modification(self, layer_types, layer_numbers, max_mod=5):
        # Calculate initial perplexity with original model weights
        inital_performance = self.calculate_model_peformance()
        print("=" * 50)
        print(f"The initial performance of the model is {inital_performance}")
        print("=" * 50)
        best_performance = inital_performance
        optimal_params = (None, None)
        mods = 0

        for layer_number in layer_numbers:
            for layer_type in layer_types:
                if mods >= max_mod and max_mod != -1:
                    return optimal_params, best_performance
                attempt = (layer_type, layer_number)
                if attempt in self.failed_attempts:
                    continue  # Skip this attempt if it has failed before

                try_update = self.update_model_reduce_layer(layer_type, layer_number)

                if not try_update:
                    continue  # Skip this attempt if it has already been modified before

                try:
                    performance = self.calculate_model_peformance()
                    if best_performance < performance:
                        best_performance = performance
                        optimal_params = (layer_type, layer_number)
                        mods = mods + 1
                        # Break out of the loop as soon as a better configuration is found
                        print("*" * 50)
                        print(
                            f"Improved performance: {best_performance} for layer {layer_type} {layer_number}. Total modifications is {mods}"
                        )
                        print("*" * 50)
                    else:
                        self.restore_model_original_layer(layer_type, layer_number)
                        self.failed_attempts.add(attempt)  # Record the failed attempt

                except NotImplementedError:
                    print("Perplexity calculation method is not implemented yet.")
                    return False, best_performance

        return optimal_params, best_performance

    def save_model(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
