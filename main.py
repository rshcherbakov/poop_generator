
import os
from diffusers import (StableDiffusionPipeline, 
                       EulerDiscreteScheduler)


def main():
    model_id = "stabilityai/stable-diffusion-2"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, 
                                                       subfolder="scheduler",
                                                       device_map="auto"
                                                       )
    pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                   scheduler=scheduler, 
                                                   device_map="auto"
                                                   )

    
    prompt = "wood log on the conveyor belt, view from the top"
    image = pipe(prompt, ).images[0]
        
    image.save("astronaut_rides_horse.png")


if __name__ == "__main__":
    main()