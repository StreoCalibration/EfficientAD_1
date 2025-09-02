import argparse
import yaml
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.lightning_model import EfficientADLightning
from src.data.provider import RealDatasetProvider, SyntheticDatasetProvider


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_provider(config: dict):
    """Create dataset provider based on config"""
    data_config = config['data']
    
    if data_config['source'] == 'synthetic':
        return SyntheticDatasetProvider(
            image_size=tuple(data_config['image_size']),
            num_samples=data_config.get('num_samples', 1000),
            train_batch_size=data_config['train_batch_size'],
            eval_batch_size=data_config['eval_batch_size']
        )
    elif data_config['source'] == 'real':
        return RealDatasetProvider(
            data_path=data_config['path'],
            category=data_config['category'],
            image_size=tuple(data_config['image_size']),
            train_batch_size=data_config['train_batch_size'],
            eval_batch_size=data_config['eval_batch_size']
        )
    else:
        raise ValueError(f"Unknown data source: {data_config['source']}")


def main():
    parser = argparse.ArgumentParser(description='Train EfficientAD model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create dataset provider
    dataset_provider = create_dataset_provider(config)
    
    # Create model
    model_config = config['model']
    trainer_config = config['trainer']
    
    model = EfficientADLightning(
        model_size=model_config['model_size'],
        learning_rate=config.get('learning_rate', 1e-4),
        st_weight=config.get('st_weight', 1.0),
        ae_weight=config.get('ae_weight', 1.0),
        penalty_weight=config.get('penalty_weight', 1.0),
        dataset_provider=dataset_provider,
        image_size=tuple(config['data']['image_size'])
    )
    
    # Create data loaders
    train_loader = dataset_provider.get_train_dataloader()
    val_loader = dataset_provider.get_val_dataloader()
    
    # Setup logging and checkpoints
    data_source = config['data']['source']
    category = config['data'].get('category', 'synthetic')
    
    logger = TensorBoardLogger(
        save_dir='results',
        name='EfficientAD',
        version=f"{category}_{data_source}"
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir + '/checkpoints',
        filename='efficient_ad-{epoch:02d}-{val/auroc:.3f}',
        monitor='val/auroc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/auroc',
        patience=10,
        mode='max',
        verbose=True
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config['max_epochs'],
        accelerator=trainer_config['accelerator'],
        devices=trainer_config.get('devices', 1),
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=1.0,
        deterministic=True
    )
    
    # Resume from checkpoint if specified
    ckpt_path = args.resume
    
    # Train model
    print(f"Starting training with config:")
    print(f"- Data source: {data_source}")
    print(f"- Category: {category}")
    print(f"- Model size: {model_config['model_size']}")
    print(f"- Max epochs: {trainer_config['max_epochs']}")
    print(f"- Accelerator: {trainer_config['accelerator']}")
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path
    )
    
    # Test model on best checkpoint
    print("\nTesting model on best checkpoint...")
    test_loader = dataset_provider.get_test_dataloader()
    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path='best'
    )
    
    print(f"\nTraining completed! Results saved to: {logger.log_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    main()