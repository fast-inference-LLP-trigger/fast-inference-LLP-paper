import tensorflow as tf
import tf2onnx


### girare prima questo poi su terminale il comando: ###
### python3 -m tf2onnx.convert --saved-model /tmp/saved_model --output "model.onnx"

# Specify the input and output paths
h5_model_path = 'float_VarTracks.h5'
onnx_model_path = 'VarTracks.onnx'

# Load the h5 model
h5_model = tf.keras.models.load_model(h5_model_path)

# Convert the h5 model to a TensorFlow SavedModel format
tf.saved_model.save(h5_model, '/tmp/saved_model')

# Convert the TensorFlow SavedModel to ONNX format
onnx_model, _ = tf2onnx.convert('/tmp/saved_model', opset=13)

# Save the ONNX model to the output path
with open(onnx_model_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(f'Successfully converted {h5_model_path} to {onnx_model_path}')

