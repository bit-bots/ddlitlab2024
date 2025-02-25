import abc

import cv2
import numpy as np

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.models import DEFAULT_IMG_SIZE, Image, Recording
from ddlitlab2024.dataset.resampling.max_rate_resampler import MaxRateResampler


class ImageConverter(Converter, abc.ABC):
    def __init__(self, resampler: MaxRateResampler) -> None:
        self.resampler = resampler

    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        models = ModelData()
        for sample in self.resampler.resample(data, relative_timestamp):
            models.images.append(self._create_image(sample.data, sample.timestamp, recording))
        return models

    @abc.abstractmethod
    def _create_image(self, data, sampling_timestamp: float, recording: Recording) -> Image:
        pass


class BitbotsImageConverter(ImageConverter):
    def __init__(self, resampler: MaxRateResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data: InputData, recording: Recording):
        img_scaling = (DEFAULT_IMG_SIZE[0] / data.image.width, DEFAULT_IMG_SIZE[1] / data.image.height)
        if recording.img_width_scaling == 0.0:
            recording.img_width_scaling = img_scaling[0]
        if recording.img_height_scaling == 0.0:
            recording.img_height_scaling = img_scaling[1]

        img_scaling_changed = (
            recording.img_width_scaling != img_scaling[0] or recording.img_height_scaling != img_scaling[1]
        )

        if img_scaling_changed:
            logger.error(
                "The image sizes changed during one recording! All images of a recording must have the same size."
            )

    def _create_image(self, data, sampling_timestamp: float, recording: Recording) -> Image:
        image = data.image
        img_array = np.frombuffer(image.data, np.uint8).reshape((image.height, image.width, 3))

        will_img_be_upscaled = recording.img_width_scaling > 1.0 or recording.img_height_scaling > 1.0
        interpolation = cv2.INTER_AREA
        if will_img_be_upscaled:
            interpolation = cv2.INTER_CUBIC

        resized_img = cv2.resize(img_array, (recording.img_width, recording.img_height), interpolation=interpolation)
        match data.encoding:
            case "rgb8":
                resized_rgb_img = resized_img
            case "bgr8":
                resized_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            case _:
                raise AssertionError(f"Unsupported image encoding: {image.encoding}")

        return Image(
            stamp=sampling_timestamp,
            recording=recording,
            image=resized_rgb_img,
        )


class BHumanImageConverter(ImageConverter):
    def __init__(self, resampler: MaxRateResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data: InputData, recording: Recording):
        upper = data.image
        lower = data.lower_image
        if upper is not None and lower is not None:
            assert upper.shape == lower.shape, "Upper and lower image must have the same shape"

        image = upper if upper is not None else lower

        img_scaling = (DEFAULT_IMG_SIZE[0] / image.shape[1], DEFAULT_IMG_SIZE[1] / image.shape[0])
        if recording.img_width_scaling == 0.0:
            recording.img_width_scaling = img_scaling[0]
        if recording.img_height_scaling == 0.0:
            recording.img_height_scaling = img_scaling[1]

        img_scaling_changed = (
            recording.img_width_scaling != img_scaling[0] or recording.img_height_scaling != img_scaling[1]
        )

        if img_scaling_changed:
            logger.error(
                "The image sizes changed during one recording! All images of a recording must have the same size."
            )

    def _create_image(self, data, sampling_timestamp: float, recording: Recording) -> Image:
        image = data.image if data.image is not None else data.lower_image
        assert image is not None, "Image must be available"

        will_img_be_upscaled = recording.img_width_scaling > 1.0 or recording.img_height_scaling > 1.0
        interpolation = cv2.INTER_AREA
        if will_img_be_upscaled:
            interpolation = cv2.INTER_CUBIC

        resized_img = cv2.resize(image, (recording.img_width, recording.img_height), interpolation=interpolation)

        resized_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        return Image(
            stamp=sampling_timestamp,
            recording=recording,
            image=resized_rgb_img,
        )
