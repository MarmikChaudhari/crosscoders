"""
Train a Crosscoder using pre-computed activations.

Activations are assumed to be stored in the directory specified by `--activation-store-dir`, organized by model and dataset:
    activations/<base-model>/<dataset>/<submodule-name>/
"""

import torch as th
import argparse
from pathlib import Path
from dictionary_learning.cache import PairedActivationCache


from dictionary_learning import CrossCoder
from dictionary_learning.dictionary import BatchTopKCrossCoder
from dictionary_learning.trainers import CrossCoderTrainer, BatchTopKCrossCoderTrainer
from dictionary_learning.training import trainSAE
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--base-model", type=str, default="mixtral-5l-active-27M")
    parser.add_argument("--instruct-model", type=str, default="tiny-gpt-27M")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--mu", type=float, default=3e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    parser.add_argument("--same-init-for-all-layers", action="store_true")
    parser.add_argument("--norm-init-scale", type=float, default=0.005)
    parser.add_argument("--init-with-transpose", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resample-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None, nargs="+")
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=["arxiv-code-stories"]
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--shared-features", type=int, default=0, help="number of features designated as shared")
    parser.add_argument("--shared-sparsity-penalty", type=float, default=None, help="λs - reduced penalty for shared features")
    parser.add_argument("--standard-sparsity-penalty", type=float, default=None, help="λf - normal penalty for standard features")
    parser.add_argument("--trainer-type", type=str, default="standard", choices=["standard", "batchtopk"], help="Type of trainer to use")
    parser.add_argument("--k", type=int, default=300, help="Number of top features to keep active (BatchTopK only)")
    parser.add_argument("--auxk-alpha", type=float, default=None, help="Weight for auxiliary loss (BatchTopK only)")
    args = parser.parse_args()

    print(f"Training args: {args}")
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    activation_store_dir = Path(args.activation_store_dir)

    base_model_dir = activation_store_dir / args.base_model
    instruct_model_dir = activation_store_dir / args.instruct_model
    caches = []
    submodule_name = f"layer_{args.layer}_out"

    for dataset in args.dataset:
        base_model_dataset = base_model_dir / dataset
        instruct_model_dataset = instruct_model_dir / dataset
        caches.append(
            PairedActivationCache(
                base_model_dataset / submodule_name,
                instruct_model_dataset / submodule_name,
            )
        )

    dataset = th.utils.data.ConcatDataset(caches)
    if args.epochs > 1:
        dataset = th.utils.data.ConcatDataset([dataset] * args.epochs)

    activation_dim = dataset[0].shape[1]
    dictionary_size = args.expansion_factor * activation_dim

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Training on device={device}.")
    
    # Configure trainer based on type
    if args.trainer_type == "batchtopk":
        trainer_class = BatchTopKCrossCoderTrainer
        dict_class = BatchTopKCrossCoder
        auxk_alpha = args.auxk_alpha or (1/32)  # Default from paper
        
        trainer_cfg = {
            "trainer": trainer_class,
            "dict_class": dict_class,
            "steps": args.max_steps or 40000,
            "activation_dim": activation_dim,
            "dict_size": dictionary_size,
            "k": args.k,
            "lr": args.lr,
            "auxk_alpha": auxk_alpha,
            "device": device,
            "warmup_steps": 1000,
            "layer": args.layer,
            "lm_name": f"{args.instruct_model}-{args.base_model}",
            "wandb_name": f"L{args.layer}-k{args.k}-auxk{auxk_alpha:.1e}-lr{args.lr:.0e}"
            + (f"-{args.run_name}" if args.run_name is not None else ""),
            "shared_features": args.shared_features,
            "shared_sparsity_penalty": args.shared_sparsity_penalty,
            "standard_sparsity_penalty": args.standard_sparsity_penalty,
            "dict_class_kwargs": {
                "code_normalization": "MIXED",
                "code_normalization_alpha_sae": 0.7,
                "code_normalization_alpha_cc": 0.3,
            },
        }
    else:  # standard CrossCoder
        trainer_class = CrossCoderTrainer
        dict_class = CrossCoder
        
        trainer_cfg = {
            "trainer": trainer_class,
            "dict_class": dict_class,
            "activation_dim": activation_dim,
            "dict_size": dictionary_size,
            "lr": args.lr,
            "resample_steps": args.resample_steps,
            "device": device,
            "warmup_steps": 1000,
            "layer": args.layer,
            "lm_name": f"{args.instruct_model}-{args.base_model}",
            "compile": False,
            "wandb_name": f"L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}"
            + (f"-{args.run_name}" if args.run_name is not None else ""),
            "l1_penalty": args.mu,
            "shared_features": args.shared_features,
            "shared_sparsity_penalty": args.shared_sparsity_penalty,
            "standard_sparsity_penalty": args.standard_sparsity_penalty,
            "dict_class_kwargs": {
                "same_init_for_all_layers": args.same_init_for_all_layers,
                "norm_init_scale": args.norm_init_scale,
                "init_with_transpose": args.init_with_transpose,
                "encoder_layers": args.encoder_layers,
                "code_normalization": "MIXED",
                "code_normalization_alpha_sae": 0.7,
                "code_normalization_alpha_cc": 0.3,
            },
            "pretrained_ae": (
                CrossCoder.from_pretrained(args.pretrained)
                if args.pretrained is not None
                else None
            ),
        }

    validation_size = 10**6
    train_dataset, validation_dataset = th.utils.data.random_split(
        dataset, [len(dataset) - validation_size, validation_size]
    )
    print(f"Training on {len(train_dataset)} token activations.")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=args.validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        save_dir="checkpoints",
        steps=args.max_steps,
        save_steps=args.validate_every_n_steps,
    )