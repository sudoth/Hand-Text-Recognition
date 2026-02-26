from dataclasses import dataclass


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Стандартный алгоритм с dp"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[m]


def cer(pred: str, truth: str) -> float:
    ref = list(truth)
    hyp = list(pred)
    if len(ref) == 0:
        return 0 if len(hyp) == 0 else 1.0
    return levenshtein_distance(hyp, ref) / len(ref)


def wer(pred: str, truth: str) -> float:
    ref = truth.split()
    hyp = pred.split()
    if len(ref) == 0:
        return 0 if len(hyp) == 0 else 1.0
    return levenshtein_distance(hyp, ref) / len(ref)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0
