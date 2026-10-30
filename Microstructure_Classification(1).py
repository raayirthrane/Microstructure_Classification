import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

# 1. Setup Dynamic Directories
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'Data_steel', 'Train_set')
test_dir = os.path.join(base_dir, 'Data_steel', 'Test_set')
single_pred_dir = os.path.join(base_dir, 'Data_steel', 'single_prediction', 'net.tif')

classes = ['Network', 'Pearlite_Spheroidite', 'Martensite', 'Pearlite']

# 2. Configure Generators with Validation Split and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2  # 80% Train, 20% Validation
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 3. Create Data Generators (Increased resolution to 128x128 for clearer microstructural features)
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    color_mode='grayscale',
    subset='training'
)

val_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    color_mode='grayscale',
    subset='validation'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    color_mode='grayscale',
    shuffle=False
)

# 4. Handle Class Imbalance
weights = compute_class_weight('balanced', classes=np.unique(train_set.classes), y=train_set.classes)
class_weight_dict = dict(enumerate(weights))

# 5. Build CNN Architecture with Batch Normalization
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[128, 128, 1]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='softmax')
])

# 6. Compile and Train
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn, metrics=['accuracy'])
cnn.summary()

# Callbacks for dynamic optimization
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

cnn.fit(
    x=train_set,
    validation_data=val_set,
    epochs=50,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_decay]
)

# 7. Detailed Evaluation on Test Set
print("\n" + "=" * 60)
print("             DETAILED TEST SET EVALUATION BY CLASS")
print("=" * 60)

test_set.reset()
predictions_prob = cnn.predict(test_set)
y_pred = np.argmax(predictions_prob, axis=-1)
y_true = test_set.classes
filenames = test_set.filenames

# Compute per-sample losses
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
y_true_one_hot = tf.one_hot(y_true, depth=len(classes)).numpy()
sample_losses = cce(y_true_one_hot, predictions_prob).numpy()

class_map = {v: k for k, v in test_set.class_indices.items()}

for cls_idx, cls_name in class_map.items():
    cls_indices = np.where(y_true == cls_idx)[0]
    total_class_images = len(cls_indices)

    if total_class_images == 0:
        print(f"\nClass: '{cls_name}' -> No test images found.")
        continue

    correct_mask = (y_pred[cls_indices] == cls_idx)
    correct_count = np.sum(correct_mask)
    incorrect_count = total_class_images - correct_count
    class_loss = np.mean(sample_losses[cls_indices])

    print(f"\n--- Class: {cls_name} ---")
    print(f"Total Test Images      : {total_class_images}")
    print(f"Accurately Predicted   : {correct_count} ({correct_count / total_class_images * 100:.2f}%)")
    print(f"Inaccurately Predicted : {incorrect_count} ({incorrect_count / total_class_images * 100:.2f}%)")
    print(f"Average Class Loss     : {class_loss:.4f}")
    print(f"Class Loss Percentage  : {class_loss * 100:.2f}%")

    if incorrect_count > 0:
        print("Misclassified File(s):")
        incorrect_indices = cls_indices[~correct_mask]
        for idx in incorrect_indices:
            file_path = filenames[idx]
            predicted_label = class_map[y_pred[idx]]
            confidence = predictions_prob[idx][y_pred[idx]] * 100
            print(f"  • File: {file_path}")
            print(f"    └─ Misclassified as: '{predicted_label}' (Confidence: {confidence:.2f}%)")
    else:
        print("All images in this class were predicted correctly!")

# 8. Overall Summary Statements (Overall Loss, Loss Percentage, and R^2 Score)
overall_loss = np.mean(sample_losses)
overall_acc = np.mean(y_pred == y_true)
overall_loss_percentage = overall_loss * 100

# R^2 Score calculation on predicted probability distribution vs one-hot targets
r2 = r2_score(y_true_one_hot, predictions_prob)

print("\n" + "=" * 60)
print(f"Overall Test Loss            : {overall_loss:.4f}")
print(f"Overall Test Loss Percentage : {overall_loss_percentage:.2f}%")
print(f"Overall Test Accuracy        : {overall_acc * 100:.2f}%")
print(f"Overall R² Score             : {r2:.4f}")
print("=" * 60)

# 9. Single Image Prediction
test_image = load_img(single_pred_dir, target_size=IMAGE_SIZE, color_mode='grayscale')
test_image = img_to_array(test_image) / 255.0
test_image = np.expand_dims(test_image, axis=0)

predictions = cnn.predict(test_image)
predicted_class_idx = np.argmax(predictions, axis=-1)[0]

labels = {v: k for k, v in train_set.class_indices.items()}
print("\nSingle Prediction Probabilities:", predictions)
print("Predicted Class:", labels[predicted_class_idx])