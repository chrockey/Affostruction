import io
from contextlib import contextmanager
import torch
import torch.distributed as dist


def read_file_dist(path):
    """
    Read the binary file distributedly.
    File is only read once by the rank 0 process and broadcasted to other processes.

    Returns:
        data (io.BytesIO): The binary data read from the file.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        size = torch.LongTensor(1).cuda()
        if dist.get_rank() == 0:
            with open(path, "rb") as f:
                data = f.read()
            data = torch.ByteTensor(
                torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
            ).cuda()
            size[0] = data.shape[0]
        dist.broadcast(size, src=0)
        if dist.get_rank() != 0:
            data = torch.ByteTensor(size[0].item()).cuda()
        dist.broadcast(data, src=0)
        data = data.cpu().numpy().tobytes()
        data = io.BytesIO(data)
        return data
    else:
        with open(path, "rb") as f:
            data = f.read()
        data = io.BytesIO(data)
        return data


@contextmanager
def local_master_first():
    """
    A context manager that ensures local master process executes first.
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
