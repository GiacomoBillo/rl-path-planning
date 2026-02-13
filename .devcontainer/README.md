# Avoid Everything Dev Container

This directory contains the development container configuration for the Avoid Everything project.

This project supports both GPU and CPU configurations. 
When opening the dev container, you can choose which configuration to use by selecting: 

+ ` cpu-container/devcontainer.json ` for CPU-only development
+ `gpu-container/devcontainer.json` for GPU development



## Prerequisites (for GPU setup)

1. **Docker** with NVIDIA Container Toolkit for GPU support
2. **VS Code** or **Cursor** with Dev Containers extension
3. **X11 forwarding** set up on your host (for GUI applications)

## Quick Start

### Using VS Code/Cursor
1. Open the project folder in VS Code/Cursor
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Select "Dev Containers: Reopen in Container"
4. Wait for the container to build and start

### Using Docker Compose directly
```bash
# Build and start the container
docker-compose up -d

# Attach to the running container
docker-compose exec avoid-everything-dev /bin/bash
```

## What's Included

- **CUDA 12.1.1** development environment
- **Python 3** with scientific computing packages
- **OMPL** (Open Motion Planning Library)
- **PyTorch** with CUDA support
- **Development tools**: cmake, git, neovim, etc.
- **GPU access** for CUDA workloads
- **X11 forwarding** for GUI applications

## Environment Variables

- `PYTHONPATH=/workspace` - Your project is automatically in the Python path
- `DISPLAY` - Forwarded from host for GUI apps
- NVIDIA GPU variables configured automatically

## Troubleshooting

### GUI Applications Not Working
Make sure X11 forwarding is enabled on your host:
```bash
xhost +local:docker
```

### GPU Not Accessible
Verify NVIDIA Container Toolkit is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Container Won't Start
Check Docker Compose logs:
```bash
docker-compose logs avoid-everything-dev
``` 