from dataclasses import dataclass


@dataclass
class GPU:
    fp16_tflops: float
    fp8_tflops: float
    mfu: float
    mem: float
    mem_bw: float  # GB/s
    nvlink_bw: float  # unidirectional GB/s
    rdma_bw: float  # unidirectional GB/s


h20 = GPU(
    fp16_tflops=148,
    fp8_tflops=296,
    mfu=0.6,
    mem=96,
    mem_bw=4096 * 0.8,
    nvlink_bw=900 * 0.8 / 2,
    rdma_bw=40,
)  # 20GB/s for 4 ibv devices, 40GB/s for 8 ibv devices

h800 = GPU(
    fp16_tflops=989,
    fp8_tflops=1979,
    mfu=0.35,
    mem=80,
    mem_bw=3430 * 0.8,
    nvlink_bw=400 * 0.8 / 2,
    rdma_bw=40,
)

gpu_map = {"H20": h20, "H800": h800}
