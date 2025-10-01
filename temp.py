import tensorflow as tf
# Check the version
print("TensorFlow version:", tf.__version__)

# Check for available GPUs (should list a 'METAL' device)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))