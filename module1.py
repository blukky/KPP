import tensorflow as tf
import tensorflow_hub as hub
import pathlib

module_selection = ("inception_v3", 229) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32 #@param {type:"integer"}

data_dir = "ranks"
val_dir = 'ranks_valid'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
val_dir = pathlib.Path(val_dir)
image_count_val = len(list(val_dir.glob('*/*.jpg')))


datagen_kwargs = dict(rescale=1./255, validation_split=.20)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=10,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)

train_data = datagen.flow_from_directory(data_dir,
                                         target_size=IMAGE_SIZE,
                                         batch_size=BATCH_SIZE,
                                         class_mode="sparse")
valid_data = datagen.flow_from_directory(val_dir,
                                         target_size=IMAGE_SIZE,
                                         batch_size=BATCH_SIZE,
                                         class_mode="sparse")

do_fine_tuning = False #@param {type:"boolean"}
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_data.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()




model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])




steps_per_epoch = image_count // BATCH_SIZE
validation_steps = image_count_val // BATCH_SIZE
hist = model.fit(
    train_data,
    epochs=15, steps_per_epoch=steps_per_epoch,
    validation_data=valid_data,
    validation_steps=validation_steps).history


plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])