import math
import numpy as np
import multiprocessing as mp
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_similarity

# Split text into sentences
def split_sentences(doc):
  sentences = doc.split('. ')
  return sentences

def create_embeddings(sentences, model):
  embeddings = model.encode(sentences)
  return embeddings

def create_similarities(embeddings):
  similarities = cosine_similarity(embeddings)
  return similarities

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))
    
def activate_similarities(similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum 
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid) 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities

def get_minmimas(activated_similarities):
  minmimas = argrelextrema(activated_similarities, np.less, order=2) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
  return minmimas

def normalize_sentence_length(sentences):
  # Get the length of each sentence
  sentece_length = [len(each) for each in sentences]
  # Determine longest outlier
  long = np.mean(sentece_length) + np.std(sentece_length) *2
  # Determine shortest outlier
  short = np.mean(sentece_length) - np.std(sentece_length) *2
  # Shorten long sentences
  text = ''
  for each in sentences:
      if len(each) > long:
          # let's replace all the commas with dots
          comma_splitted = each.replace(',', '.')
      else:
          text+= f'{each}. '
  sentences = text.split('. ')
  # Now let's concatenate short ones
  text = ''
  for each in sentences:
      if len(each) < short:
          text+= f'{each} '
      else:
          text+= f'{each}. '
  sentences = text.split('. ')
  return sentences

def get_paragraphs(sentences, minmimas):
  #Get the order number of the sentences which are in splitting points
  split_points = [each for each in minmimas[0]]
  # Create empty string
  text = []
  paragraph = ''
  for num,each in enumerate(sentences):
      # Check if sentence is a minima (splitting point)
      if num in split_points:
        text.append(paragraph)
        paragraph = f'{each}. '
          # # If it is than add a dot to the end of the sentence and a paragraph before it.
          # text+=f'\n\n {each}. '
      else:
        # If it is a normal sentence just add a dot to the end and keep adding sentences.
        paragraph+=f'{each}. '
        # text[i]+=f'{each}. '
  return text

def pooled_paragraph_summary_pipeline(text, model, summarizer, p=10):
  pool = mp.Pool(int(mp.cpu_count()/4))
  sentences = split_sentences(text)
  embeddings = create_embeddings(sentences, model)
  similarities = create_similarities(embeddings)
  activated_similarities = activate_similarities(similarities, p_size=p)
  minmimas = get_minmimas(activated_similarities)
  norm_sentences = normalize_sentence_length(sentences)
  paragraphs = get_paragraphs(norm_sentences, minmimas)
  para_sum = pool.map(summarizer, [x for x in paragraphs])
  joined_summary = ' '.join([x[0]['summary_text'] for x in para_sum])
  return joined_summary

