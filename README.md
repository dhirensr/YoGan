# YoGan
GAN For Yoga Text Description to Image 

![YoGan description image](https://github.com/dhirensr/YoGan/blob/master/github_repo_yogan.png)
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

* Python 3



## Installing


```
pip3 install requirements.txt
```
After all the needed packages are install one can run the jupyter notebook directly to infer from all the models.

## To Run Demo and generate images
```
cd YoGan/demo
jupyter-notebook yogan_demo.ipynb
```

Note: generated images are saved in YoGan/demo/data/outputs 
You can edit these paths in this notebook.


## Changing the dataset and few hyperparameters

In the file dcgan_train.py we have the option to change the files of the dataset and also few more parameters like image width,height and channels of the image, the noise vector size ,number of epochs and batch size.

## Training the model
We have uploaded all the models in the directory final_models. If one wants to train from scratch and generate the model then one can use dcgan_train.py

## Built With

* [Tensorflow](https://www.tensorflow.org/) - for backend with Keras
* [Scikit Image](https://scikit-image.org/docs/dev/api/skimage.html) - Image processing library
* [Keras](https://keras.io/) - Deep Learning framework


## Contributing

1. Fork it (<https://github.com/dhirensr/YoGan/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Authors

* **Dhiren Serai** - *Author* - [Github Profile](https://github.com/dhirensr)
* **Shashank Salian** - *Author* - [Github Profile](https://github.com/shashank3110)


See also the list of [contributors](https://github.com/dhirensr/YoGan/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [GANCLS](https://medium.com/datadriveninvestor/text-to-image-synthesis-6e5de1bf86ec)
* [DcGan](https://medium.com/datadriveninvestor/deep-convolutional-generative-adversarial-networks-dcgans-3176238b5a3d)
