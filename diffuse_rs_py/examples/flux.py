from diffuse_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.ModelId("black-forest-labs/FLUX.1-dev"))

image_bytes = pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=50, guidance_scale=3.5
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
