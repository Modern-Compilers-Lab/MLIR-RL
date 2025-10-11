from utils.singleton import Singleton
import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(metaclass=Singleton):
    """Logger using TensorBoard for training metrics and results."""

    def __init__(self, log_dir: str, run_name: str, tags: list[str] = None):
        """
        Args:
            log_dir (str): Base directory for logs (e.g. "logs").
            run_name (str): Custom run name (instead of auto run_0).
            tags (list[str], optional): Tags or metadata for this run.
        """
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # Save tags to a file for reproducibility
        if tags:
            with open(os.path.join(self.run_dir, "tags.txt"), "w") as f:
                f.write("\n".join(tags) + "\n")

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under a main tag (TensorBoard grouping)."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def flush(self):
        """Flush events to disk (useful if crashing)."""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

