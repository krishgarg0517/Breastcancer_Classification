import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np, os, cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Dice Loss ---
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Data Loader with Augmentation ---
def load_data(folder, img_size=256):
    X, Y = [], []
    for subfolder in ["benign", "malignant", "normal"]:
        path = os.path.join(folder, subfolder)
        for file in os.listdir(path):
            if "_mask" not in file and file.endswith(".png"):
                img_path = os.path.join(path, file)
                mask_path = img_path.replace(".png", "_mask.png")
                if not os.path.exists(mask_path):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    continue
                img = cv2.resize(img, (img_size, img_size)) / 255.0
                mask = cv2.resize(mask, (img_size, img_size)) / 255.0
                X.append(img)
                Y.append(mask)
    X = np.expand_dims(np.array(X), -1)
    Y = np.expand_dims(np.array(Y), -1)
    return X, Y

X, Y = load_data(r"C:data/Dataset_BUSI_with_GT", img_size=256)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Augment training data ---
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(X_train)

# --- Deeper U-Net ---
def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    c5 = conv_block(p4, 512)

    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4]); c6 = conv_block(u6, 256)
    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3]); c7 = conv_block(u7, 128)
    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2]); c8 = conv_block(u8, 64)
    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1]); c9 = conv_block(u9, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    model = models.Model(inputs, outputs)
    return model

model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=bce_dice_loss,
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=8),
    validation_data=(X_test, Y_test),
    epochs=40,
    callbacks=callbacks
)

model.save("tumor_unet_optimized.h5")
