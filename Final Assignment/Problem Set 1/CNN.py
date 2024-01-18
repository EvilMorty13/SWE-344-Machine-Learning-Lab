import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to the dataset
train_dir = os.path.join(os.getcwd(), "Dataset\\Archive\\train")
test_dir = os.path.join(os.getcwd(), "Dataset\\Archive\\test")
val_dir = os.path.join(os.getcwd(), "Dataset\\Archive\\val")

# Define hyperparameters
input_shape = (224, 224, 3)
batch_size = 32
epochs = 10

# Create a CNN model
model = models.Sequential()
model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create data generators with augmentation for training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(val_dir,
                                                        target_size=input_shape[:2],
                                                        batch_size=batch_size,
                                                        class_mode='sparse')

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size,
                    verbose=1)

# Evaluate the model on the test set
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=input_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='sparse')

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size, verbose=1)

# Save all data in a single text file
results_dir = os.path.join(os.getcwd(), "training_results")
os.makedirs(results_dir, exist_ok=True)

result_txt_path = os.path.join(results_dir, "training_results.txt")
with open(result_txt_path, 'w') as f:
    f.write("Epoch\tTrain Acc\tVal Acc\n")
    for epoch, acc, val_acc in zip(range(1, epochs+1), history.history['accuracy'], history.history['val_accuracy']):
        f.write(f"{epoch}\t{acc:.4f}\t{val_acc:.4f}\n")
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n")

# Save the model in Keras native format
model.save(os.path.join(results_dir, 'image_classification_model.keras'))
