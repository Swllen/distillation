"""Keep the upstream logging API using maintained PyTorch TensorBoard."""
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir, flush_secs=2):
        self.writer = SummaryWriter(log_dir=logdir, flush_secs=flush_secs)

    def log_value(self, name, value, step):
        self.writer.add_scalar(name, float(value), step)

    def close(self):
        self.writer.close()
