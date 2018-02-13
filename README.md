# keras-gan-models

Some generative adversarial network models that I studied 

# Usage

### Deep Convolution GAN

Below is the [sample codes](keras_gan_models/demo/dcgan_train.py) to train the DCGan on a set of pokemon sample images

```python
from keras_gan_models.library.dcgan import DCGan
from keras_gan_models.library.utility.image_loader import load_and_scale_images


def main():
    image_dir_path = './data/images'
    model_dir_path = './models'

    img_width = 32
    img_height = 32
    img_channels = 3

    gan = DCGan()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 200

    images = load_and_scale_images(image_dir_path, '.png', img_width, img_height)

    batch_size = 4
    epochs = 2000
    gan.fit(model_dir_path, images=images, batch_size=batch_size, epochs=epochs,
            snapshot_dir_path='./data/outputs', snapshot_interval=100)



if __name__ == '__main__':
    main()
```

Below is the [sample codes](keras_gan_models/demo/dcgan_generate.py) on how to load the trained DCGan model to generate
3 new pokemon samples:

```python
from keras_gan_models.library.dcgan import DCGan


def main():
    model_dir_path = './models'

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image = gan.generate_image()
        image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '.png')
```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 