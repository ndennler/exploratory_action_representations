import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

# Define a function to create the grid
def create_image_grid(images, grid_size=(4, 7), img_size=(512, 512), space_size=(300, 100)):
    # Create a blank image for the grid
    grid_width = grid_size[1] * img_size[0] + 2 * space_size[0]
    grid_height = grid_size[0] * img_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Calculate positions for images
    positions = [(i % grid_size[1], i // grid_size[1]) for i in range(grid_size[0] * grid_size[1])]
    
    # Paste images into the grid with spaces
    for idx, pos in enumerate(positions):
        
        image = images[idx]
        image = ImageOps.fit(image, img_size, Image.Resampling.LANCZOS)  # Resize image to fit cell
        
        space = 0
        if pos[0] > 0:
            space += space_size[0]
        if pos[0] > 3:
            space += space_size[0]

        x_pos = pos[0] * img_size[0] + space
        y_pos = pos[1] * img_size[1]
        grid_image.paste(image, (x_pos, y_pos))
    
    return grid_image


def closest_vector(reference_vector, array_of_vectors):
    # Normalize the reference vector
    ref_norm = reference_vector / np.linalg.norm(reference_vector)
    
    # Normalize the array of vectors
    norms = np.linalg.norm(array_of_vectors, axis=1)
    normalized_vectors = array_of_vectors / norms[:, np.newaxis]
    
    # Compute the cosine similarities
    similarities = np.dot(normalized_vectors, ref_norm)
    
    # Find the index of the maximum similarity
    sorted_indices = np.argsort(-similarities)  # Sort indices by descending order of similarity
    second_closest_index = sorted_indices[1]

    # print(similarities[second_closest_index])
    
    return second_closest_index





TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

impaths = pd.read_csv('../data/all_data.csv') 


ids = []
#find closest images to a given image
# good idles 139, 326,
# good searching 226
# good item 536, 4436
for signal, id in zip(['idle', 'searching', 'has_information', 'has_item'], [3265,886,108,536]):
    ids.append(id)

    for technique in ['random', 'autoencoder', 'VAE', 'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']:
        embeds = np.load(f'../data/embeds/visual&independent&raw&{technique}&all_signals&8.npy')
        
        options = embeds[:, TASK_INDEX_MAPPING[signal], :]
        query = embeds[id, TASK_INDEX_MAPPING[signal]]

        ids.append(closest_vector(query, options))





# Load your images
image_paths = ['../data/visual/vis/' + impaths.query(f'id == {id} and type == "Video"')['file'].values[0].replace('.mp4', '.jpg') for id in ids]
images = [Image.open(img_path) for img_path in image_paths]

# Create the image grid
grid_image = create_image_grid(images)

# Save or display the grid image
grid_image.show()  # To display the image
grid_image.save('image_grid.jpg')  # To save the image