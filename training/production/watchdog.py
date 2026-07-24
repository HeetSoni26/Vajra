import math
import torch
import logging

logger = logging.getLogger(__name__)


class NumericalStabilityWatchdog:
    """Monitors gradients and loss for NaNs/Infs during training."""

    def __init__(self, nan_detection: bool = True, inf_detection: bool = True):
        self.nan_detection = nan_detection
        self.inf_detection = inf_detection
        self.skip_requested = False

    def check_loss(self, loss_val: float) -> bool:
        """Returns True if loss is stable, False if NaN/Inf."""
        if self.nan_detection and math.isnan(loss_val):
            logger.warning("Watchdog detected NaN loss!")
            return False
        if self.inf_detection and math.isinf(loss_val):
            logger.warning("Watchdog detected Inf loss!")
            return False
        return True

    def check_gradients(self, model: torch.nn.Module) -> bool:
        """Checks for NaN/Inf in gradients. Returns True if stable."""
        self.skip_requested = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if self.nan_detection and torch.isnan(param.grad).any():
                    logger.warning(f"Watchdog detected NaN gradient in {name}")
                    self.skip_requested = True
                    return False
                if self.inf_detection and torch.isinf(param.grad).any():
                    logger.warning(f"Watchdog detected Inf gradient in {name}")
                    self.skip_requested = True
                    return False
        return True
