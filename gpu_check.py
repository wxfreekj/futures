"""RTX 5070 Ti / PyTorch CUDA diagnostics (safe: no market orders or trading)."""
import platform
import sys
import time


def main():
    print("Python:", sys.version.split()[0], "| OS:", platform.platform())
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Use https://pytorch.org/get-started/locally/")
        raise SystemExit(1)

    print("PyTorch:", torch.__version__, "| Built for CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No CUDA device accessible: verify nvidia-smi and install a compatible CUDA-enabled PyTorch wheel.")
        raise SystemExit(2)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print("GPU name:", props.name)
    print("Compute capability:", ".".join(map(str, torch.cuda.get_device_capability(0))))
    print("GPU memory GiB:", round(props.total_memory / 2**30, 2))

    # 2048² is a modest operation to confirm native GPU kernels work.
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    print("Matrix multiplication:", tuple(c.shape), "| seconds:", round(time.perf_counter() - started, 4))
    print("Finite output:", bool(torch.isfinite(c).all().item()))
    print("Success: a real PyTorch CUDA operation ran on the selected GPU.")


if __name__ == "__main__":
    main()