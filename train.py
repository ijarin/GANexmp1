# Imports

import numpy as np
from numpy import random
#%matplotlib inline
from util import view_samples, sigmoid, Discriminator,Generator

# Examples of faces
faces = [np.array([1,0,0,1]),
         np.array([0.9,0.1,0.2,0.8]),
         np.array([0.9,0.2,0.1,0.8]),
         np.array([0.8,0.1,0.2,0.9]),
         np.array([0.8,0.2,0.1,0.9])]

f1,f2=view_samples(faces,1,4)
#save face image
f1.savefig('face.png')
# Examples of noisy images
noise = [np.random.randn(2,2) for i in range(20)]
def generate_random_image():
    return [np.random.random(), np.random.random(), np.random.random(), np.random.random()]

n1,n2 = view_samples(noise, 4,5)
#save noisy image
n1.savefig('noise.png')

# Set random seed
np.random.seed(42)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# The GAN
D = Discriminator(learning_rate)
G = Generator(learning_rate)

# For the error plot
errors_discriminator = []
errors_generator = []

for epoch in range(epochs):

    for face in faces:
        # Update the discriminator weights from the real face
        D.update_from_image(face,learning_rate)

        # Pick a random number to generate a fake face
        z = random.rand()

        # Calculate the discriminator error
        errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))

        # Calculate the generator error
        errors_generator.append(G.error(z, D))

        # Build a fake face
        noise = G.forward(z)

        # Update the discriminator weights from the fake face
        D.update_from_noise(noise,learning_rate)

        # Update the generator weights from the fake face
        G.update(z, D,learning_rate)
        #print("training Done")
#Generated Images
generated_images = []
for i in range(4):
    z = random.random()
    generated_image = G.forward(z)
    generated_images.append(generated_image)
g1,g2=view_samples(generated_images,1,4)

#save generated image files
g1.savefig('GenImg.png')
print("execution completed")