# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load the EfficientNet model
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create a new model on top
model = Sequential([
  base_model,
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create an ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load the FaceForensics dataset
train_data = datagen.flow_from_directory('path_to_dataset', target_size=(224, 224), class_mode='binary', subset='training')
val_data = datagen.flow_from_directory('path_to_dataset', target_size=(224, 224), class_mode='binary', subset='validation')

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the model
model.save('deepfake_detection_model.h5')

# Testing
# Load the model
model = tf.keras.models.load_model('deepfake_detection_model.h5')

# Load the test data

test_data = datagen.flow_from_directory('path_to_test_data', target_size=(224, 224), class_mode='binary')


# Evaluate the model

model.evaluate(test_data)

# Make predictions

predictions = model.predict(test_data)

# Show statistics

print(predictions)

# Accuracy
accuracy = (predictions == test_data.classes).mean()
print('Accuracy:', accuracy)
