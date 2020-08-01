import numpy as np


def time_for_early_stopping(val_losses: list, look_behind: int):
    if len(val_losses) < look_behind:
        return False

    last_epoch_loss = val_losses[-1]
    avg_epoch_loss = np.average(val_losses[-look_behind:])

    # Stop training if the progress in last epoch is less than 5% of average losses
    return avg_epoch_loss - last_epoch_loss < 0.05 * avg_epoch_loss
