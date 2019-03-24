# Sentiment Classifier

This classifier is trained using two separate models to investigate the performance in its accuracy of prediction. For full documentation, visit the [Jupyter Notebook walkthrough](https://github.com/cindywang3299/twitter-sentiment-analyzer/tree/master/models) and [notes](https://github.com/cindywang3299/twitter-sentiment-analyzer/tree/master/deliverables) during the development process. The trained weights have been applied using Flask, using a new dataset of movie reviews to validate the training outcomes.

##Datasets
- [IMDB Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Sentiment140](http://ai.stanford.edu/~amaas/data/sentiment/), available from the dataset provided by Stanford.

##Procedures
- Start a virtual environment and install `requirements.txt`
- Build the sentiment classifier
  - The dataset needs to be installed manually:
  ```
  wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  tar xvzf aclImdb_v1.tar.gz
  ```
  - `sentiment_analysis.ipynb` will then place the dataset under directory `./data/aclImdb` and do all the heavy lifting.
- Write the `app.py` which is the API application that will be deployed
- Update requirements.txt as you write the code
- Test the API

##Testing the model
1. Create a virtual environment called .mais-env: `python -m venv .mais-env`
2. Activate the environment so we are using the fresh, new python environment.
UNIX or MacOS: `source .mais-env/bin/activate`
Windows: `.mais-env\Scripts\activate`
3. Use the requirements.txt file to install all the dependencies in your newly created environment: `pip install -r requirements.txt`
4. Add a new kernel to Jupyter Notebook: `python -m ipykernel install --user --name .mais-env --display-name "Python (MAIS-202)"`
5. Finally, open the `jupyter notebook` of the sentiment_analysis.

##Testing the API
1. Fork this repository.
2. Go to the `movieclassifier` directory and run the Flask API locally using the command `python app.py`.
3. Testing the API at http://127.0.0.1:5000/
