import torch as t
from collections import namedtuple
from typing import Optional

from ..dictionary import BatchTopKSAE
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


class BatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = BatchTopKSAE,
        pretrained_ae: Optional[BatchTopKSAE] = None,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "BatchTopKSAE",
        activation_mean: Optional[t.Tensor] = None,
        activation_std: Optional[t.Tensor] = None,
        target_rms: float = 1.0,
        encoder_init_norm: float = 1.0,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dim,
                dict_size,
                k,
                activation_mean=activation_mean,
                activation_std=activation_std,
                target_rms=target_rms,
                encoder_init_norm=encoder_init_norm,
            )
        else:
            self.ae = pretrained_ae

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.encoder_init_norm = encoder_init_norm

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(
        self,
        x,
        step=None,
        logging=False,
        use_threshold=False,
        normalize_activations=True,
        inplace_normalize=True,
        **kwargs,
    ):
        x = (
            self.ae.normalize_activations(x, inplace=inplace_normalize)
            if normalize_activations
            else x
        )
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x,
            return_active=True,
            use_threshold=use_threshold,
            normalize_activations=False,
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f, denormalize_activations=False)

        e = x - x_hat

        self.effective_l0 = self.k

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            assert f.shape == (x.shape[0], self.ae.dict_size)
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "deads": ~did_fire,
                    "threshold": self.ae.threshold.item(),
                    "rms_norm": t.sqrt((x.pow(2).sum(-1)).mean()).item(),
                },
            )

    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        x = self.ae.normalize_activations(
            x,
            inplace=True,  # Normalize inplace to avoid copying the activations during training
        )
        loss = self.loss(x, step=step, normalize_activations=False)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "BatchTopKTrainer",
            "dict_class": "BatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "encoder_init_norm": self.encoder_init_norm,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
