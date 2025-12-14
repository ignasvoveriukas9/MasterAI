
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU active: {gpus}")
    except RuntimeError as e:
        print(e)
        
from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
#print("⚙️  Mixed precision training enabled")

import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Path setup
base_dir = "/home/ignas/VU/Master/AI/fgvc-aircraft-2013b"
images_dir = os.path.join(base_dir, "data", "images")
labels_file = os.path.join(base_dir, "data", "images_variant_trainval.txt")

# Read label mappings
image_paths, labels = [], []
with open(labels_file, "r") as f:
    for line in f:
        img_name, variant = line.strip().split(" ", 1)
        image_paths.append(os.path.join(images_dir, img_name + ".jpg"))
        labels.append(variant)

print(f"Found {len(image_paths)} images")

# Build label -> integer mapping
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
num_classes = len(label_to_index)
print(f"{num_classes} unique aircraft variants")

# Convert string labels to indices
y = [label_to_index[lbl] for lbl in labels]

# Create tf.data.Dataset
path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(y)
dataset = tf.data.Dataset.zip((path_ds, label_ds))

imgsize = 348

def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img[:-20, :, :]  # crop bottom 20 pixels
    img = tf.image.resize(img, [imgsize, imgsize])
    #img = tf.image.resize(img, [384, 384])
    img = preprocess_input(img)  # use ResNet preprocessing
    return img, label


dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=False)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),          # random horizontal flip
    layers.RandomRotation(0.1),               
#    layers.RandomZoom(0.1),            
#    layers.RandomContrast(0.1),     
#    layers.RandomTranslation(0.1, 0.1),  
], name="data_augmentation")

# Split
train_size = int(0.8 * len(image_paths))
train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

batch_size = 16
train_ds = train_ds.batch(batch_size).map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Dataset ready")

# Convert label back to readable text
index_to_label = {v: k for k, v in label_to_index.items()}

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Spatial transformer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout

class Localization(tf.keras.layers.Layer):
    def __init__(self):
        super(Localization, self).__init__()
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv1 = tf.keras.layers.Conv2D(20, [5, 5], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(20, [5, 5], activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(20, activation='relu')
        self.fc2 = tf.keras.layers.Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer='zeros')

    def build(self, input_shape):
        print("Building Localization Network with input shape:", input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta = self.fc2(x)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta

class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    def build(self, input_shape):
        print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''        
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
            
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates
    
    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])
                
            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]
                
            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)
                            
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
                        
        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])
        

#base_model = ResNet50(
#    weights='imagenet',        # Load pretrained weights
#    include_top=False,         # Remove the final classification layer
#    input_shape=(imgsize, imgsize, 3)
#)

#base_model.trainable = False  # Freeze the convolutional base (for initial training)

num_classes = len(label_to_index)

from tensorflow.keras import layers

#model = models.Sequential([
#    base_model,
#    layers.GlobalAveragePooling2D(),
#   layers.Dense(512, activation='relu'),
#   layers.Dropout(0.5),
#    layers.Dense(num_classes, activation='softmax')
#])

def build_stn_resnet(imgsize, num_classes):

    # ---- Input ----
    inputs = tf.keras.Input(shape=(imgsize, imgsize, 3))

    # ---- Spatial Transformer ----
    theta = Localization()(inputs)  
    x = BilinearInterpolation(height=imgsize, width=imgsize)([inputs, theta])

    # ---- ResNet50 ----
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(imgsize, imgsize, 3)
    )

    base_model.trainable = False  # freeze initially

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)

    # ---- Final classifier ----
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model
    
model, base_model = build_stn_resnet(imgsize, num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=17
)

model.save('/home/ignas/VU/Master/AI/fgvc_stn_resnet50_finetuned.h5')

import numpy as np

#for images, labels in val_ds.take(1):
#    preds = model.predict(images)
#    pred_labels = np.argmax(preds, axis=1)
#    for i in range(3):
#        plt.imshow((images[i] + 1) / 2)
#        plt.title(f"Pred: {index_to_label[pred_labels[i]]}\nTrue: {index_to_label[int(labels[i])]}")
#        plt.axis('off')
#        plt.show()
        

#Test

import tensorflow as tf

from tensorflow.keras.models import load_model

model = load_model(
    "/home/ignas/VU/Master/AI/fgvc_stn_resnet50_finetuned.h5",
    custom_objects={
        "Localization": Localization,
        "BilinearInterpolation": BilinearInterpolation
    }
)
print("Model loaded ")

import os

base_dir = "/home/ignas/VU/Master/AI/fgvc-aircraft-2013b"
images_dir = os.path.join(base_dir, "data", "images")
test_labels_file = os.path.join(base_dir, "data", "images_variant_test.txt")

image_paths_test, labels_test = [], []
with open(test_labels_file, "r") as f:
    for line in f:
        img_name, variant = line.strip().split(" ", 1)
        image_paths_test.append(os.path.join(images_dir, img_name + ".jpg"))
        labels_test.append(variant)

# Map string labels to integer indices (use same mapping as training!)
y_test = [label_to_index.get(lbl, -1) for lbl in labels_test]

from tensorflow.keras.applications.resnet50 import preprocess_input

def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img[:-20, :, :]       # remove bottom banner
    img = tf.image.resize(img, [imgsize, imgsize])
    img = preprocess_input(img)  # ResNet preprocessing
    return img, label

path_ds_test = tf.data.Dataset.from_tensor_slices(image_paths_test)
label_ds_test = tf.data.Dataset.from_tensor_slices(y_test)
test_ds = tf.data.Dataset.zip((path_ds_test, label_ds_test))

batch_size = 32
test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

# Gather all predictions and true labels
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute multi-class metrics (macro = average over classes)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("✅ Final evaluation metrics:")
print(f"   Precision (macro): {precision:.4f}")
print(f"   Recall (macro):    {recall:.4f}")
print(f"   F1-score (macro):  {f1:.4f}")

index_to_label = {v: k for k, v in label_to_index.items()}


import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def predict_single_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img[:-20, :, :]        # crop banner
    img = tf.image.resize(img, [imgsize, imgsize])
    img = preprocess_input(img)
    img = tf.expand_dims(img, 0) # add batch dimension
    preds = model.predict(img)
    predicted_label = np.argmax(preds, axis=1)[0]
    return index_to_label[predicted_label]

# Example:
test_image_path = image_paths_test[0]
predicted_variant = predict_single_image(test_image_path)
print("Predicted aircraft variant:", predicted_variant)
