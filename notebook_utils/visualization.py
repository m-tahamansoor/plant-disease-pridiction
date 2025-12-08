# notebook_utils/visualize.py
import src
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss curves.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for an input image.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(sample_img, heatmap, title_suffix=""):
    """
    Displays the original image and the image overlaid with the Grad-CAM heatmap.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img)
    plt.title("Actual Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sample_img)
    plt.imshow(heatmap, alpha=0.6, cmap='jet')
    plt.title(f"AI Attention Map (Grad-CAM) {title_suffix}")
    plt.axis('off')
    plt.show()
    

# Example usage (can be run in a separate notebook or interactive session):
if __name__ == '__main__':
    from src.train import main
    from src.config import LAST_CONV_LAYER_NAME
    
    # Run the main training to get the model and history
    history, trained_model = main()
    
    # Plot the history
    plot_training_history(history)
    
    # Example for Grad-CAM (Requires a data generator)
    from src.data_loader import create_generators
    _, val_generator, _ = create_generators()
    
    # Get a sample image
    images, _ = next(val_generator)
    sample_img = images[0]
    sample_img_array = np.expand_dims(sample_img, axis=0)
    
    # Run Grad-CAM
    heatmap = make_gradcam_heatmap(sample_img_array, trained_model, LAST_CONV_LAYER_NAME)
    
    # Display
    display_gradcam(sample_img, heatmap)