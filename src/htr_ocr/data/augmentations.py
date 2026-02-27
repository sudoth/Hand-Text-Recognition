import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


from torchvision.transforms.functional import affine
import torchvision.transforms as T

from PIL import Image


def _rand_uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


@dataclass(frozen=True)
class RandomShear:

    max_degrees: float = 7.0
    p: float = 0.5
    fill: int = 255

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        deg = float(_rand_uniform(-self.max_degrees, self.max_degrees))

        return affine(
            img,
            angle=0.0,
            translate=(0, 0),
            scale=1.0,
            shear=(deg, 0.0),
            fill=self.fill,
        )


@dataclass(frozen=True)
class RandomRotate:
    """Небольшое вращение для имитации наклона почерка"""

    max_degrees: float = 3.0
    p: float = 0.5
    fill: int = 255

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        angle = float(_rand_uniform(-self.max_degrees, self.max_degrees))
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=self.fill)


@dataclass(frozen=True)
class RandomPerspective:
    """Изменение перспективы"""

    distortion_scale: float = 0.2
    p: float = 0.5
    fill: int = 255

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        t = T.RandomPerspective(distortion_scale=float(self.distortion_scale), p=1.0, fill=self.fill)
        return t(img)


@dataclass(frozen=True)
class RandomStretch:
    """Горизонтальное растяжение"""

    max_factor: float = 0.15
    p: float = 0.5

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        factor = float(_rand_uniform(1.0 - self.max_factor, 1.0 + self.max_factor))
        new_w = max(1, int(round(w * factor)))
        return img.resize((new_w, h), resample=Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class RandomDistort:
    """Легкое искажение с использованием вертикальной сетки"""

    max_shift_px: int = 6
    num_stripes: int = 12
    p: float = 0.5
    fill: int = 255

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        if w < 4 or h < 4:
            return img

        n = max(2, int(self.num_stripes))
        max_s = max(1, int(self.max_shift_px))

        period = _rand_uniform(w * 0.6, w * 1.4)
        phase = _rand_uniform(0.0, 2 * math.pi)

        def shift_at(x: float) -> float:
            return max_s * math.sin((2 * math.pi * x / period) + phase)

        mesh = []
        xs = [int(round(i * w / n)) for i in range(n + 1)]
        for i in range(n):
            x0, x1 = xs[i], xs[i + 1]
            if x1 <= x0:
                continue
            xc = 0.5 * (x0 + x1)
            s0 = shift_at(x0)
            s1 = shift_at(x1)

            box = (x0, 0, x1, h)

            quad = (
                x0,
                -s0,
                x1,
                -s1,
                x1,
                h - s1,
                x0,
                h - s0,
            )
            mesh.append((box, quad))

        return img.transform(
            img.size,
            Image.Transform.MESH,
            mesh,
            resample=Image.Resampling.BILINEAR,
            fillcolor=self.fill,
        )


@dataclass(frozen=True)
class RandomElastic:
    """Эластичное преобразование"""

    alpha: float = 40.0
    sigma: float = 6.0
    p: float = 0.5
    fill: int = 255

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        t = T.ElasticTransform(alpha=float(self.alpha), sigma=float(self.sigma), fill=self.fill)
        return t(img)


@dataclass(frozen=True)
class RandomOneOf:
    """Применяем ровно одну аугментация для читаемости."""

    transforms: Sequence[Callable[[Image.Image], Image.Image]]
    p_total: float = 0.5

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.transforms:
            return img
        if random.random() > self.p_total:
            return img
        t = random.choice(list(self.transforms))
        return t(img)
