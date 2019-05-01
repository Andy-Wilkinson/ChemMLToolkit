import tensorflow as tf


class VariableScheduler(tf.keras.callbacks.Callback):
    """Schedules an arbitary variable during training.
    Arguments:
        variable: The variable to modify the value of.
        schedule:  A function that takes an epoch index (integer, indexed
            from 0) and current variable value as input and returns a new
            value to assign to the variable as output.
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, variable, schedule, verbose=0):
        super(VariableScheduler, self).__init__()
        self.variable = variable
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        value = self.variable.read_value()
        value = self.schedule(epoch, value)
        self.variable.assign(value)
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: VariableScheduler assigning '
                  f'variable {self.variable.name} to {value}.')
