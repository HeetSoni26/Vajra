from pydantic import BaseModel


class DDPConfig(BaseModel):
    enabled: bool = False
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    find_unused_parameters: bool = False
    static_graph: bool = False
    broadcast_buffers: bool = True
    timeout_minutes: int = 30
