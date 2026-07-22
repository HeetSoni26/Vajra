class MultiNodeLauncher:
    """
    Abstraction for future multi-node launcher.
    Currently inactive as per requirements.
    """
    
    def __init__(self, config):
        self.config = config
        
    def setup_cluster(self):
        """Discovers nodes and sets up rendezvous for multi-node training."""
        if not self.config.enabled:
            return
        raise NotImplementedError("Multi-node training is a future milestone.")

class CommunicationAbstraction:
    """
    Abstracts future multi-node communication bounds (e.g. FSDP / DeepSpeed).
    """
    
    @staticmethod
    def sync_gradients():
        pass
