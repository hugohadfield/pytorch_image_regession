"""
This script generates a sample dataset for the image regression task.
Specifically we set up the problem of estimating the 2D vector that corresponds
to an arrow drawn on an image. In this script we sythentically generate the images
and write the corresponding targets to a csv file.
"""

from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


def synthesise_image(
    image_size: Tuple[int, int], 
    vec: Tuple[float, float], 
    origin: Tuple[int, int]
) -> np.ndarray:
    """
    This function synthesises an image of a 2D vector on a black background.
    The vector is drawn in white and the background is black.
    """
    # Create the image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Draw the vector
    tip = (int(origin[0] + vec[0]), int(origin[1] + vec[1]))
    tip_color = (0, 0, 255)
    line_color = (255, 255, 255)
    image = cv2.arrowedLine(
        image, origin, tip, tip_color, 2
    )
    image = cv2.line(
        image, origin, tip, line_color, 2
    )
    return image
    

def synthesise_dataset(
    subfolder_name: str, 
    num_images: int,
    image_size: Tuple[int, int], 
    rng_seed: int
) -> pd.DataFrame:
    """
    Synthesise a dataset of images and targets. The images are saved to the subdirectory
    and the targets are saved to a csv file. The function returns the dataframe containing
    the targets.
    """
    # Set the random seed
    np.random.seed(rng_seed)

    # Create the dataframe
    df = pd.DataFrame(columns=['image_path', 'x', 'y'])

    # Create the images and targets
    for i in range(num_images):
        # Generate the target
        target = (np.random.randn(), np.random.randn())
        target = (target[0]/np.linalg.norm(target), target[1]/np.linalg.norm(target))

        # Generate the image
        arrow_length = 0.8*(min(image_size)//2)
        image = synthesise_image(
            image_size, 
            (arrow_length*target[0], arrow_length*target[1]),
            (min(image_size)//2, min(image_size)//2)
        )

        # Save the image
        dataset_filename = f'example_dataset/{subfolder_name}/images/image_{i}.png'
        Path(dataset_filename).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(dataset_filename, image)

        # Add the target to the dataframe
        df.loc[i] = [dataset_filename, target[0], target[1]]

    # Save the dataframe
    df.to_csv(f'example_dataset/{subfolder_name}.csv', index=False)
    return df


if __name__ == '__main__':
    synthesise_dataset('train', 1000, (220, 200), 42)
    synthesise_dataset('test', 100, (220, 200), 43)
