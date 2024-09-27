# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, File
from refiners.solutions import BoxSegmenter
from PIL import Image

import torch
import io


class Predictor(BasePredictor):

    def setup(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.segmenter = BoxSegmenter(device="cpu")
        self.segmenter.device = device
        self.segmenter.model = self.segmenter.model.to(device=self.segmenter.device)


    def predict(self, image_file: File = Input(description="Input image")) -> Path:
        image = Image.open(image_file)
        image = image.rotate(-90.0, expand=True)
        box = [0, 0, image.width, image.height]

        img, masked_rgb = self.process(image, box)
        masked_rgb = masked_rgb.rotate(90.0, expand=True)

        img_byte_arr = io.BytesIO()
        masked_rgb.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        output_path = "output.png"
        with open(output_path, "wb") as f:
            f.write(img_byte_arr.getvalue())

        return Path(output_path)


    def process(self, image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, Image.Image]:
        if image.width > 2048 or image.height > 2048:
            orig_res = max(image.width, image.height)
            image.thumbnail((2048, 2048))
            x0, y0, x1, y1 = (int(x * 2048 / orig_res) for x in box)
            box = (x0, y0, x1, y1)

        mask = self.segmenter(image, box)
        masked_alpha = self.apply_mask(image, mask)
        return (image, masked_alpha)


    def apply_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")

        result = Image.new("RGBA", image.size)
        result.paste(image, (0, 0), mask)
        return result
