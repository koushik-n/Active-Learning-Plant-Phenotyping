# Active Learning for Image-based plant phenotyping
This repository contains the code used to run the deep active learning experiments detailed in our paper "How useful is Active Learning for Image-based plant Phenotyping?"
## Dependencies
In order to run our code, you'll need these main packages:

- [Python>=3.5](https://www.python.org/)
- [Numpy>=1.14.3](https://numpy.org/)
- [Scipy>=1.0.0](https://www.scipy.org/)
- [TensorFlow>=1.5](https://www.tensorflow.org/)
- [Keras>=2.2](https://keras.io/)
## Running the code
The code for running uncertainity methods is in AL_uncertainity.py and for coreset is AL_coreset.py. This code was modified from the code for [Discriminative Active Learning](https://github.com/dsgissin/DiscriminativeActiveLearning) paper.
```
python3 main.py <experiment_index> <batch_size> <initial_size> <iterations> <output_path> -idx <indices_path> -wip <weights_path> -dp <data_path> -lp <labels_path> -gpu <gpus>
```
- experiment_index: an integer detailing the number of experiment (since usually many are run in parallel and combined later).
- batch_size: the size of the batch of examples to be labeled in every iteration.
- initial_size: the amount of labeled examples to start the experiment with (chosen randomly).
- iteration: the amount of active learning iterations to run in the experiment.
- method: a string for the name of the query strategy to be used in the experiment ("Random", "LC" (Least Confidence"), Entropy, BALD).
- output_path: the path of the folder where the experiment data and results will be saved.
- idx: a path to the folder with the pickle file containing the initial labeled example indices for the experiment (optional).
- wip: a path to file containing initial model weights (usually the weights after training using the first batch of random sampling. This is optional)
- dp: a path to complete dataset in numpy format
- lp: a path to labels in numpy format
- gpu: the gpus to use for training the models.


