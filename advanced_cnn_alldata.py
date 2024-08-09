import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def combine_datasets(base_dir, combined_dir):
    create_dir(combined_dir)
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            combined_category_path = os.path.join(combined_dir, category)
            create_dir(combined_category_path)
            for dataset_type in ['default', 'real_world']:
                dataset_path = os.path.join(category_path, dataset_type)
                if os.path.exists(dataset_path):
                    for image in os.listdir(dataset_path):
                        shutil.copy(os.path.join(dataset_path, image), os.path.join(combined_category_path, image))

def split_dataset(combined_dir, output_dir, test_size=0.15, val_size=0.15):
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(output_dir, split))

    for category in os.listdir(combined_dir):
        category_path = os.path.join(combined_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            train_images, temp_images = train_test_split(images, test_size=(test_size + val_size), random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=(test_size / (test_size + val_size)), random_state=42)

            train_output_dir = os.path.join(output_dir, 'train', category)
            create_dir(train_output_dir)
            for image in train_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(train_output_dir, image))

            val_output_dir = os.path.join(output_dir, 'val', category)
            create_dir(val_output_dir)
            for image in val_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(val_output_dir, image))

            test_output_dir = os.path.join(output_dir, 'test', category)
            create_dir(test_output_dir)
            for image in test_images:
                shutil.copy(os.path.join(category_path, image), os.path.join(test_output_dir, image))

def create_data_generators(output_dir):
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        os.path.join(output_dir, 'train'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = datagen.flow_from_directory(
        os.path.join(output_dir, 'val'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(output_dir, 'test'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def build_model(base_model_name, num_classes):
    if base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    elif base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    else:
        raise ValueError("Invalid base model name")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, title, results_dir):
    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training and Validation Accuracy - {title}')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss - {title}')

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{title}_training_history.png')
    plt.savefig(plot_path)
    plt.close()

def evaluate_and_plot_confusion_matrix(model, test_generator, title, results_dir):
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_acc:.4f}')
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {title}')
    plot_path = os.path.join(results_dir, f'{title}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Classification Report - {title}')
    target_names = list(test_generator.class_indices.keys())
    report = classification_report(test_generator.classes, y_pred, target_names=target_names)
    report_path = os.path.join(results_dir, f'{title}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

def main():
    base_dir = 'RHWC_DATA/images/images'  # Replace with the path to your dataset
    combined_dir = 'RHWC_DATA/combined_images'  # Combined directory
    output_dir = 'RHWC_DATA/processed_images'  # Output directory
    results_dir = 'RHWC_DATA/results'  # Results directory

    create_dir(results_dir)
    combine_datasets(base_dir, combined_dir)
    split_dataset(combined_dir, output_dir)
    
    base_model_names = ['InceptionV3', 'VGG16', 'ResNet50']
    
    for base_model_name in base_model_names:
        print(f'Training using {base_model_name} model...')
        train_generator, val_generator, test_generator = create_data_generators(output_dir)
        model_save_path = os.path.join(results_dir, f'waste_classification_model_combined_{base_model_name}.keras')
        
        model = build_model(base_model_name, train_generator.num_classes)
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        
        history = model.fit(
            train_generator,
            epochs=50,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stop]
        )
        
        plot_training_history(history, f'Combined - {base_model_name}', results_dir)
        evaluate_and_plot_confusion_matrix(model, test_generator, f'Combined - {base_model_name}', results_dir)
        
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
