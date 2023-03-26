from keras.callbacks import TensorBoard
from tensorflow.summary import create_file_writer as FileWriter
import tensorflow as tf

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        self._write_logs(stats, step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.writer.flush()
