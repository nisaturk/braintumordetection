import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Nisa/Desktop/Brain Tumor Detection/scans', 
    image_size=(224, 224),  # Resizing to 224x224 because ResNet50 expects this input size
    batch_size=32,
    label_mode='binary',
    shuffle=True
)

# Splitting into training and validation sets
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freezing all the layers in the base model because they won't be initially trained

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compiled the model with a super low LR for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)
base_model.trainable = True # Defreeze

# Freezing some of the early layers and only fine-tuning the last few ones
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile the model with a very low LR again
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

history_fine_tune = model.fit(train_dataset, validation_data=val_dataset, epochs=5) # Fine-tuning the model

# Evaluating the fine-tuned model
val_loss, val_acc = model.evaluate(val_dataset)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

plt.plot(history_fine_tune.history['accuracy'], label='train accuracy')
plt.plot(history_fine_tune.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_fine_tune.history['loss'], label='train loss')
plt.plot(history_fine_tune.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()