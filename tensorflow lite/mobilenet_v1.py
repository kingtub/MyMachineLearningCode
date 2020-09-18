import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobilenet_v1_1.0_224.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input_details: ', input_details)
print('output_details: ', output_details)


f = open("labels.txt")
lines = f.readlines()

bbimg = cv2.imread("cat.jpg")

bbresult = cv2.resize(bbimg, dsize=(input_details[0]['shape'][2], input_details[0]['shape'][1]))
cv2.imshow("r", bbresult)
cv2.waitKey()
cv2.destroyAllWindows()

input_data = bbresult[np.newaxis, :, :, :].astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)
print('recognition result is: ', lines[output_data[0].argmax()])