import genesis as gs

# Initialize Genesis on the selected CUDA device.
gs.init(backend=gs.cuda, logging_level="warning")

DEVICE = gs.device
