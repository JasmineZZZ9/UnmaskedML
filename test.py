# import tensorflow as tf
# import numpy as np

# print(tf.add(1, 2))
# print(tf.add([1, 2], [3, 4]))
# print(tf.square(5))
# print(tf.reduce_sum([1, 2, 3]))

# print(tf.square(2) + tf.square(3))

# x = tf.matmul([[1]], [[2, 3]])
# print(x)
# print(x.shape)
# print(x.dtype)

# print("=================================")

# ndarray = np.ones([3, 3])

# print("Tensorflow operations convert numpy arrays to Tensors automatically")
# tensor = tf.multiply(ndarray, 42)
# print(tensor)

# print("And NumPy operations convert Tensors to numpy arrays automatically")
# print(np.add(tensor, 1))

# print("The .numpy() method explicitly converts a Tensor to a numpy array")
# print(tensor.numpy())

# print("=================================")

# x = tf.random.uniform([3, 3])
# print("Is there a GPU available: ")
# print(tf.config.list_physical_devices("GPU"))

# print("Is the Tensor on GPU #0: ")
# print(x.device.endswith("GPU:0"))

# import tempfile
# ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# # Create a CSV file
# _, filename = tempfile.mkstemp()

# with open(filename, 'w') as f:
#     f.write("""Line 1
# Line 2
# Line 3
#   """)

# ds_file = tf.data.TextLineDataset(filename)

# ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

# ds_file = ds_file.batch(2)

# print('Elements of ds_tensors:')
# for x in ds_tensors:
#     print(x)

# print('\nElements in ds_file:')
# for x in ds_file:
#     print(x)
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
