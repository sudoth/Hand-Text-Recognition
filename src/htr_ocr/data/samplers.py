from dataclasses import dataclass
import random
from typing import Iterator, Sequence


@dataclass
class BucketBatchSampler:
    """
    Сортируем все батчи по длинам и отдаем в батч похожие.
    Это уменьшит паддинг.
    """

    lengths: Sequence[int]
    batch_size: int
    shuffle_batches: bool = True
    seed: int = 42
    drop_last: bool = False

    def __iter__(self) -> Iterator[list[int]]:
        bs = int(self.batch_size)
        if bs <= 0:
            raise ValueError("Empty batch")

        order = sorted(range(len(self.lengths)), key=lambda i: int(self.lengths[i]))
        batches = [order[i : i + bs] for i in range(0, len(order), bs)]

        if self.drop_last and batches and len(batches[-1]) < bs:
            batches = batches[:-1]

        if self.shuffle_batches:
            rnd = random.Random(int(self.seed))
            rnd.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self) -> int:
        n = len(self.lengths)
        bs = int(self.batch_size)
        if bs <= 0:
            return 0
        if self.drop_last:
            return n // bs
        return (n + bs - 1) // bs