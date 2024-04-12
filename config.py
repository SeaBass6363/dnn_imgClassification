from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:

    batch_size: int # Batch size
    num_epochs: int # Number of epochs to train
    model_folder: str # Folder where to save the checkpoints
    model_basename: str # Basename of the checkpoint files 
    preload: str # Preload weights from a previous checkpoint
    local_rank: int = -1 # LOCAL_RANK assigned by torchrun
    global_rank: int = -1 # RANK assigned by torchrun

def get_default_config() -> ModelConfig:

    return ModelConfig(
        batch_size=4,
        num_epochs=5,
        model_folder="training_data",
        model_basename="model_{0:02d}.pt",
        preload="latest",
    )

def get_file_path(config: ModelConfig, epoch: str) -> str:
    model_folder = config.model_folder
    model_basename = config.model_basename
    model_filename = model_basename.format(epoch)
    return str(Path('.') / model_folder / model_filename)

def get_latest_file_path(config: ModelConfig) -> str:
    model_folder = config.model_folder
    model_basename = config.model_basename
    # Check all files in the model folder
    model_files = Path(model_folder).glob(f"*.pt")
    # Sort by epoch number (ascending order)
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    if len(model_files) == 0:
        return None
    # Get the last one
    model_filename = model_files[-1]
    return str(model_filename)