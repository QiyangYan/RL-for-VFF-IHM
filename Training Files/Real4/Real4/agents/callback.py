from pytorch_tabnet.callbacks import Callback
import os
import matplotlib.pyplot as plt
import numpy as np

class ModelCheckpoint(Callback):
    def __init__(self, save_path, save_interval=10, monitor='valid_logloss', mode='min'):
        super().__init__()
        self.save_path = save_path
        self.save_interval = save_interval
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.best_epoch = 0
        self.history = {'epoch': [], 'train_accuracy': [], 'valid_accuracy': [], 'train_logloss': [],
                        'valid_logloss': []}

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'.")

        # Initialize the plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax1.set_title('Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax2.set_title('Log Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Log Loss')
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        current = logs[self.monitor]

        # Save history
        self.history['epoch'].append(epoch)
        self.history['train_accuracy'].append(logs['train_accuracy'])
        self.history['valid_accuracy'].append(logs['valid_accuracy'])
        self.history['train_logloss'].append(logs['train_logloss'])
        self.history['valid_logloss'].append(logs['valid_logloss'])

        # Save best model
        if self.best is None or (self.mode == 'min' and current < self.best) or (
                self.mode == 'max' and current > self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_accuracy = logs['valid_accuracy']
            self.trainer.save_model(os.path.join(self.save_path, 'best_model'))

        # Save model every save_interval epochs
        if (epoch + 1) % self.save_interval == 0:
            self.trainer.save_model(os.path.join(self.save_path, f'model_epoch_{epoch + 1}'))

        # Update the plots
        self.ax1.clear()
        self.ax1.set_title('Accuracy')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.plot(self.history['epoch'], self.history['train_accuracy'], label='Train Accuracy', color='blue')
        self.ax1.plot(self.history['epoch'], self.history['valid_accuracy'], label='Valid Accuracy', color='orange')
        self.ax1.legend()

        # Calculate variance
        if len(self.history['train_logloss']) > 1:
            train_logloss_var = np.var(self.history['train_logloss'])
            valid_logloss_var = np.var(self.history['valid_logloss'])
        else:
            train_logloss_var = valid_logloss_var = 0

        self.ax2.clear()
        self.ax2.set_title('Log Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Log Loss')
        self.ax2.plot(self.history['epoch'], self.history['train_logloss'], label='Train Log Loss', color='blue')
        self.ax2.plot(self.history['epoch'], self.history['valid_logloss'], label='Valid Log Loss', color='orange')
        self.ax2.fill_between(self.history['epoch'],
                              np.array(self.history['train_logloss']) - np.sqrt(train_logloss_var),
                              np.array(self.history['train_logloss']) + np.sqrt(train_logloss_var),
                              color='blue', alpha=0.2)
        self.ax2.fill_between(self.history['epoch'],
                              np.array(self.history['valid_logloss']) - np.sqrt(valid_logloss_var),
                              np.array(self.history['valid_logloss']) + np.sqrt(valid_logloss_var),
                              color='orange', alpha=0.2)
        self.ax2.legend()

        # Redraw the canvas
        self.fig.canvas.draw()
        # Flush the GUI events
        self.fig.canvas.flush_events()

    def get_best_accuracy(self):
        return self.best_accuracy

    def get_history(self):
        return self.history

