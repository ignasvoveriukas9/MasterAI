
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.initializers import glorot_normal
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input



import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU active: {gpus}")
    except RuntimeError as e:
        print(e)
        
#from tensorflow.keras import mixed_precision
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

imgsize = 224

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
#    layers.RandomRotation(0.1),            
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

import matplotlib.pyplot as plt

# Convert label back to readable text
index_to_label = {v: k for k, v in label_to_index.items()}

import tensorflow as tf
from tensorflow.keras import layers, models


def dot_product(x):

    return keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 

"""
Calculate signed square root

@param
x -> a tensor

"""

def signed_sqrt(x):

    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

"""
Calculate L2-norm

@param
x -> a tensor

"""

def L2_norm(x, axis=-1):

    return keras.backend.l2_normalize(x, axis=axis)


def bilinear_cnn_model(input_shape=(imgsize, imgsize, 3), num_classes=100):

    inputs = tf.keras.Input(shape=input_shape)
    
    # Two parallel VGG16 backbones
    model_detector = keras.applications.vgg16.VGG16(
                            input_tensor=inputs, 
                            include_top=False,
                            weights='imagenet')
    
    model_detector2 = keras.applications.vgg16.VGG16(
                            input_tensor=inputs, 
                            include_top=False,
                            weights='imagenet')
    
    model_detector2 = keras.models.Sequential(layers=model_detector2.layers)
    
    for i, layer in enumerate(model_detector2.layers):
        layer._name = layer.name  +"_second"
    
    model2 = keras.models.Model(inputs=[inputs], outputs = [model_detector2.layers[-1].output])
    
    x = model_detector.layers[17].output
    z = model_detector.layers[17].output_shape
    y = model2.layers[17].output
    
    print(model_detector.summary())
    
    print(model2.summary())
    
#    rehape to (batch_size, total_pixels, filter_size)
    x = layers.Reshape([z[1] * z[2] , z[-1]])(x)   
    y = layers.Reshape([z[1] * z[2] , z[-1]])(y)
    
    #   outer products of x, y
    x = layers.Lambda(dot_product)([x, y])
    
#   rehape to (batch_size, filter_size_vgg_last_layer*filter_vgg_last_layer)
    x = layers.Reshape([z[-1]*z[-1]])(x)
        
#   signed_sqrt
    x = layers.Lambda(signed_sqrt)(x)
        
#   L2_norm
    x = layers.Lambda(L2_norm)(x)
    
#   FC-Layer

    initializer = tf.keras.initializers.GlorotNormal()
            
    x = layers.Dense(units=num_classes, 
                   kernel_regularizer=keras.regularizers.l2(0.0),
                   kernel_initializer=initializer)(x)

    tensor_prediction = layers.Activation("softmax")(x)

    model_bilinear = models.Model(inputs=[inputs],
                                outputs=[tensor_prediction])
    
    
#   Freeze VGG layers
    for layer in model_detector.layers:
        layer.trainable = False
        
    sgd = keras.optimizers.SGD(learning_rate=1.0, momentum=0.9, nesterov=True)

    model_bilinear.compile(loss="sparse_categorical_crossentropy", 
                           optimizer=sgd,
                           metrics=['accuracy'])

    model_bilinear.summary()
    
    return model_bilinear


model = bilinear_cnn_model(input_shape=(imgsize, imgsize, 3), num_classes=len(label_to_index))


sgd = keras.optimizers.SGD(learning_rate=0.25, momentum=0.9, nesterov=True)
                               
model.compile(loss="sparse_categorical_crossentropy", 
	optimizer=sgd,
	metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=13
)

for layer in model.layers:
    layer.trainable = True

sgd = keras.optimizers.legacy.SGD(lr=2e-3, 
                               decay=1e-9,
                               momentum=0.9)    
    
model.compile(loss="sparse_categorical_crossentropy", 
                           optimizer=sgd,
                           metrics=['accuracy'])

print('train')

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

model.save('/home/ignas/VU/Master/AI/fgvc_bilinear_vgg_finetuned.keras')

import numpy as np

"""Test"""

import tensorflow as tf

model = tf.keras.models.load_model('/home/ignas/VU/Master/AI/fgvc_bilinear_vgg_finetuned.keras', safe_mode=False)
print("Model loaded")

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
print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

# Gather all predictions and true labels
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images,verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute multi-class metrics (macro = average over classes)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("Final evaluation metrics:")
print(f"   Precision (macro): {precision:.4f}")
print(f"   Recall (macro):    {recall:.4f}")
print(f"   F1-score (macro):  {f1:.4f}")

index_to_label = {v: k for k, v in label_to_index.items()}

import tensorflow as tf
import numpy as np

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
