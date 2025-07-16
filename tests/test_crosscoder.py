import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th
from dictionary_learning.trainers.crosscoder import CrossCoderTrainer



def test_crosscoder_trainer_loss(device):

    trainer = CrossCoderTrainer(
        activation_dim=64,
        dict_size=256,
        num_layers=2,
        layer=6,
        lm_name="test_model",
        device=device,
        seed=42,
    )
    
    x = th.randn(4, 2, 64, device=device)
    print(f'input shape: {x.shape}')
    
    loss_log = trainer.loss(x, return_deads=True, logging=True)
    
    losses = loss_log.losses
    print(f'l2 loss: {losses["l2_loss"]}')
    print(f'mse loss: {losses["mse_loss"]}')
    print(f'sparsity loss: {losses["sparsity_loss"]}')
    print(f'loss: {losses["loss"]}')
    print(f'deads: {losses["deads"]}')
    print(f'losses: {losses}')
    
    assert losses['l2_loss'] > 0
    assert losses['mse_loss'] > 0
    assert losses['sparsity_loss'] >= 0
    assert losses['loss'] > 0
    


def test_crosscoder_trainer_resample_neurons(device):
    """Test the resample_neurons method."""
    trainer = CrossCoderTrainer(
        activation_dim=64,
        dict_size=256,
        num_layers=2,
        layer=6,
        lm_name="test_model",
        resample_steps=1000,
        device=device,
        seed=42,
    )
    
    # Create dummy activations
    activations = th.randn(4, 2, 64, device=device)
    
    # Create a mask for dead neurons (let's say first 10 are dead)
    deads = th.zeros(256, dtype=th.bool, device=device)
    deads[:10] = True
    
    # Store original weights for comparison
    original_encoder_weights = trainer.ae.encoder.weight.clone()
    original_decoder_weights = trainer.ae.decoder.weight.clone()
    original_encoder_bias = trainer.ae.encoder.bias.clone()
    
    # Resample neurons
    trainer.resample_neurons(deads, activations)
    
    # Check that dead neurons' weights were changed
    assert not th.allclose(
        trainer.ae.encoder.weight[:, :, :10], 
        original_encoder_weights[:, :, :10]
    )
    assert not th.allclose(
        trainer.ae.decoder.weight[:, :10, :], 
        original_decoder_weights[:, :10, :]
    )
    assert not th.allclose(
        trainer.ae.encoder.bias[:10], 
        original_encoder_bias[:10]
    )
    
    # Check that living neurons' weights were unchanged
    assert th.allclose(
        trainer.ae.encoder.weight[:, :, 10:], 
        original_encoder_weights[:, :, 10:]
    )
    assert th.allclose(
        trainer.ae.decoder.weight[:, 10:, :], 
        original_decoder_weights[:, 10:, :]
    )
    assert th.allclose(
        trainer.ae.encoder.bias[10:], 
        original_encoder_bias[10:]
    )


def test_crosscoder_trainer_update(device):
    """test the update method."""
    trainer = CrossCoderTrainer(
        activation_dim=64,
        dict_size=256,
        num_layers=2,
        layer=6,
        lm_name="test_model",
        device=device,
        seed=42,
    )
    
    # Create dummy activations
    activations = th.randn(4, 2, 64, device=device)
    
    # Perform update
    step = 5
    trainer.update(step, activations)

    
    current_lr = trainer.optimizer.param_groups[0]['lr']
    print(f'current lr: {current_lr}')



def test_crosscoder_trainer_update_with_resampling(device):
    """Test update method with resampling enabled."""
    trainer = CrossCoderTrainer(
        activation_dim=64,
        dict_size=256,
        num_layers=2,
        layer=6,
        lm_name="test_model",
        resample_steps=10,  # Small resample_steps for testing
        device=device,
        seed=42,
    )
    
    # Create dummy activations
    activations = th.randn(4, 2, 64, device=device)
    
    # Perform updates for several steps
    for step in range(15):
        trainer.update(step, activations)

    print(f'update method runs with resampling enabled')



test_crosscoder_trainer_loss(device='cpu')