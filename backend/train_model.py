import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import json
import os

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load and preprocess the data
def prepare_data():
    # Read labels
    df = pd.read_csv('C:/Users/lenovo/OneDrive/Desktop/dog/dog-breed-identification/labels.csv')

    
    # Create directories if they don't exist
    os.makedirs('model', exist_ok=True)
    
    # Save class names
    class_names = sorted(df['breed'].unique())
    with open('model/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Create label mapping
    breed_to_label = {breed: idx for idx, breed in enumerate(class_names)}
    df['label'] = df['breed'].map(breed_to_label)
    
    # Image data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Create train and validation generators
    train_generator = train_datagen.flow_from_directory(
        '../dog-breed-identification/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        '../dog-breed-identification/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator, len(class_names)

def create_model(num_classes):
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the pre-trained layers
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    # Prepare data
    train_generator, validation_generator, num_classes = prepare_data()
    
    # Create and compile model
    model = create_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    x_batch, y_batch = next(train_generator)
    print("✅ X batch shape:", x_batch.shape)
    print("✅ Y batch shape:", y_batch.shape)
    print("✅ Y sample (first one-hot vector):", y_batch[0])

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )

    
    # Save the model
    model.save('model/dog_breed_model.h5')
    
    return history

if __name__ == "__main__":
    history = train_model()
    print("Model training completed and saved!") 