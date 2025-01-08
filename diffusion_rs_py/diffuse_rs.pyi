from dataclasses import dataclass
from enum import Enum

@dataclass
class ModelDType(Enum):
    """
    DType for the model.
    Note: When using `Auto`, fallback pattern is: BF16 -> F16 -> F32
    """

    Auto = 0
    BF16 = 1
    F16 = 2
    F32 = 2

@dataclass
class Offloading(Enum):
    """
    Offloading settings for the model.
    """

    Full = 0

@dataclass
class ModelSource(Enum):
    """
    Source of the model: either a Hugging Face model ID (including local paths) or a DDUF file
    """
    @dataclass
    class ModelId:
        model_id: str

    @dataclass
    class DdufFile:
        file: str

@dataclass
class DiffusionGenerationParams:
    """
    Generation parameters for diffusion models
    """

    height: int
    width: int
    num_steps: int
    guidance_scale: float

class Pipeline:
    def __init__(
        self,
        source: ModelSource,
        silent: bool = False,
        token: str | None = None,
        revision: str | None = None,
        offloading: Offloading | None = None,
        ModelDType: ModelDType = ModelDType.Auto,
    ) -> None:
        """
        Load a model.

        - `source`: the source of the model
        - `silent`: silent loading, defaults to `False`.
        - `token`: specifies a literal Hugging Face token for accessing gated models.
        - `revision`: specifies a specific Hugging Face model revision, otherwise the default is used.
        - `token_source` specifies where to load the HF token from.
        - `offloading`: offloading setting for the model.
        - `dtype`: dtype selection for the model. The default is to use an automatic strategy with a fallback pattern: BF16 -> F16 -> F32
        """
        ...

    def forward(
        self,
        prompts: list[str],
        params: DiffusionGenerationParams,
    ) -> list[bytes]:
        """
        Execute the diffusion model on the given batch of prompts.

        Image data is returned as bytes objects and is in the order of the prompts
        """
