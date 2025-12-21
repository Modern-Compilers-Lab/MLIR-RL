import torch
import platform
import subprocess

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        # CUDA version from torch
        print(f"PyTorch CUDA Version: {torch.version.cuda}")

        # Information about each GPU
        for i in range(num_gpus):
            print(f"\nDevice {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Device Properties: {torch.cuda.get_device_properties(i)}")

        # Driver version
        print(f"Driver Version: {torch.version.cuda}")
        
        # Run nvidia-smi command to get additional GPU details
        try:
            nvidia_smi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
            print(f"\nNVIDIA-SMI Output:\n{nvidia_smi_output.stdout}")
        except FileNotFoundError:
            print("nvidia-smi command not found. Ensure NVIDIA drivers are installed correctly.")
        
    else:
        print("No CUDA-enabled GPU detected.")

    # Print system and Python environment information
    print("\nSystem and Environment Information:")
    print(f"  Operating System: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  PyTorch Version: {torch.__version__}")

if __name__ == "__main__":
    check_cuda()