# cnn_model.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Image size
img_size = 64

# Data generators
train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    'data/Dataset/train',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_data.flow_from_directory(
    'data/Dataset/test',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("model/skin_disease_model.h5", monitor='val_accuracy', save_best_only=True)

# Train model
model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[checkpoint])
