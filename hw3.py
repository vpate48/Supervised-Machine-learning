import numpy as np
from scipy import stats
import pandas as pd
#%matplotlib inline
#import matplotlib.pyplot as plt
#import seaborn as sns
import nltk
import sklearn
import re
import string
#from IPython.display import display, Latex, Markdown
#%%

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Verify that the following commands work for you, before moving on.

lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')

#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    token=[]    
    text = re.split('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    text = " ".join(text)
    text = text.lower()
    text = text.replace('-', ' ')
    text = text.replace("'s", ' ')
    text = text.replace("'", '')
    
    pun = string.punctuation
    for a in range(len(pun)):
        text = text.replace(pun[a],' ')
                
    bs =nltk.word_tokenize(text)
    tmap = nltk.pos_tag(bs)
    
    for x in tmap :
        if x[1][0] in posMapping:
            token.append(lemmatizer.lemmatize(x[0],posMapping[x[1][0]]))
        else:
            token.append(lemmatizer.lemmatize(x[0],'n'))
    return token
    
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    df['text'] = df['text'].apply(process)
    return df
    
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    tt =  processed_tweets['text']
    cv = TfidfVectorizer(input = 'tt',tokenizer = list,min_df=2, stop_words=stop_words, lowercase = False)
#     x =cv.fit_transform(tt)
    #tfidf = cv.get_feature_names()
    tfidf = cv.fit_transform(tt)
    return (cv,tfidf)

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    a =[]
    for text in processed_tweets['screen_name']:
        if ('realDonaldTrump' in text) or ('mike_pence' in text) or ('GOP' in text):
            a.append(0)
        else:
            a.append(1)
    return np.array(a)
#%%
class MajorityLabelClassifier():
  """
  A classifier that predicts the mode of training labels
  """
  def __init__(self):
    """
    Initialize
    """
    self.mode = None
  def fit(self, X, y):
    """
    Implement fit by taking training data X and their labels y and finding the mode of y
    """
    #self.mode = max(y)    
    self.mode = stats.mode(y)[0]
  def predict(self, X):
    """
    Implement to give the mode of training labels as a prediction for each data instance in X
    return labels
    """
    preds = []
    for i in X:
        preds.append(self.mode)
    
    return np.array(preds)

#%%
from sklearn import svm
from sklearn.svm import SVC

def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.classes.SVC: classifier learnt from data
    """
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_train: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    return classifier.score(X_validation, y_validation)

#%%
kf = sklearn.model_selection.KFold(n_splits=4, random_state=1, shuffle=True)

def best_model_selection(kf, X, y):
    
    """
      Select the kernel giving best results using k-fold cross-validation.
      Other parameters should be left default.
      Input:
        kf (sklearn.model_selection.KFold): kf object defined above
        X (scipy.sparse.csr.csr_matrix): training data
        y (array(int)): training labels
      Return:
        best_kernel (string)
    """
    accuracies = []
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        classifier = learn_classifier(X_train, y_train, kernel=kernel)
        accuracy = evaluate_classifier(classifier, X_test, y_test)
        accuracies.append(accuracy)
    
    return kernels[np.argmax(accuracies)]
      # Use the documentation of KFold cross-validation to split ..
      # training data and test data from create_features() and create_labels()
      # call learn_classifer() using training split of kth fold
      # evaluate on the test split of kth fold
      # record avg accuracies and determine best model (kernel)
  #return best kernel as string


def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    processed_unlabeled_tweets = process_all(unlabeled_tweets)
    X_TesT = tfidf.transform(processed_unlabeled_tweets['text'])
    return classifier.predict(X_TesT)