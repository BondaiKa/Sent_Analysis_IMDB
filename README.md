# IMDB sentiment analysis
We created different neural netowrks and trained on IMDB dataset

# Repository description :scroll:
Models folder consist of diffrent models that can be used to classify IMDB review  


# Testing
All completed works, that was done to check tlsm network, are saved in `lstm_notebooks` folder  
If you want to test models you can use code below:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.imdb import get_word_index

MODEL_NAME = '' # add model name here
MODEL_FOLDER_PATH = "lstm_notebooks"
model_path = f"{MODEL_FOLDER_PATH}_{MODEL_NAME}"


model = tf.keras.models.load_model(model_path)
sentence = "This imdb word so beatiful from this guys. I think we should get them the highest mark!"

global word_index
word_index = get_word_index()
word_index = { word:i for (word, i) in word_index.items()}

def prepare_input(sentence:str)->np.ndarray:
    arr =  np.array([word_index.get(word,0) for word in sentence.split()])
    return np.concatenate([np.zeros(shape=(250 - arr.shape[0])),arr]).reshape(1,250)

sentence = prepare_input(sentence)[0][0]
print(f"Positive, probability: {res}" if res > 0.5 else f"Bad, probability: {probability}")

```

# Participants :alien:
- Karim Safiullin - karim.safiullin@tu-ilmenau.de  
- Anastasia - anastasiia.volgina@tu-ilmenau.de  
- Aleksandr Dmitriev - aleksandr.dmitriev@tu-ilmenau.de  
- Aleksandr Mashanin - aleksandr.mashanin@tu-ilmenau.de  