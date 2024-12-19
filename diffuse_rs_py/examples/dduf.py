from diffuse_rs import DiffusionGenerationParams, ModelSource, Pipeline

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-schnell-Q4-bnb.dduf"))

pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=4, guidance_scale=0.0
    ),
)
