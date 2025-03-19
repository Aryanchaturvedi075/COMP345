from nltk.corpus.reader.tagged import TaggedCorpusView
from sklearn.utils.extmath import randomized_svd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import Counter
from typing import Callable
import numpy as np
import re

def top_k_unigrams(tweets: list[str], stop_words: list[str], k: int) -> dict[str, int]:
    regex = re.compile(r"^[a-z#].*")
    stop_words = set(stop_words)
    
    unigram_list = [
        word.lower()
        for tweet in tweets
        for word in tweet.split()
        if regex.match(word) and word not in stop_words
    ]

    top_k_words = Counter(unigram_list)
    return top_k_words if k == -1 else dict(top_k_words.most_common(k))

### TODO: Fix this function
def context_word_frequencies(tweets: list[str], stop_words: list[str], context_size: int, frequent_unigrams) -> dict[(str, str), int]:
    # Convert to set for O(1) lookups
    frequent_unigrams = set(frequent_unigrams) if isinstance(frequent_unigrams, list) else set(frequent_unigrams.keys())
    context_pairs = []
    
    for tweet in tweets:
        # Use numpy array for faster slicing
        tokens = np.array(tweet.lower().split())
        n = len(tokens)
        
        # Create all possible context pairs efficiently
        for i in range(n):
            word1 = tokens[i]
            # Calculate context window boundaries
            start, end = max(0, i - context_size), min(n, i + context_size + 1)
            context = tokens[start:end]
            
            # Filter context words that are in frequent_unigrams
            valid_context = [w for w in context if w in frequent_unigrams and w != word1] # frequent_unigrams is a subset of top_k_words
            context_pairs.extend((word1, word2) for word2 in valid_context)
    
    context_counter = Counter(context_pairs)
    return context_counter

### TODO: Fix this function
def pmi(word1: str, word2: str, unigram_counter: dict[str, int], context_counter: dict[(str, str), int]) -> float:
    total_unigrams = float(sum(unigram_counter.values()))
    total_bigrams = float(sum(context_counter.values()))
    
    # Get the counts (with pseudo-count = 1 if not observed)
    count_w1 = float(unigram_counter.get(word1, 1))
    count_w2 = float(unigram_counter.get(word2, 1))
    count_w1_w2 = float(context_counter.get((word1, word2), 1))
    
    p_w1 = count_w1 / total_unigrams
    p_w2 = count_w2 / total_unigrams
    p_w1_w2 = count_w1_w2 / total_bigrams
    
    pmi = np.log2(p_w1_w2 / (p_w1 * p_w2))
    return pmi


def build_word_vector(word1: str, frequent_unigrams, unigram_counter: dict[str, int], context_counter: dict[(str, str), int]) -> dict[str, float]:
    frequent_unigrams = set(frequent_unigrams) if isinstance(frequent_unigrams, list) else set(frequent_unigrams.keys())
    context_set = set(context_counter.keys())
    word_vector = {}

    for word2 in frequent_unigrams:
        word_vector[word2] = float(0) if (word1, word2) not in context_set else pmi(word1, word2, unigram_counter, context_counter)
    
    return word_vector


def get_top_k_dimensions(word1_vector, k):
    sorted_items = sorted(word1_vector.items(), key=lambda x: x[1], reverse=True)
    top_k_dimensions = dict(sorted_items[:k])
    return top_k_dimensions

### TODO: Fix this function
def get_cosine_similarity(word1_vector: dict[str, float], word2_vector: dict[str, float]) -> float:
    # Convert dictionaries to numpy arrays
    vec1 = np.array([word1_vector.get(word) for word in word1_vector.keys()])
    vec2 = np.array([word2_vector.get(word) for word in word2_vector.keys()])
    
    # Use numpy's optimized operations
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    cosine_sim_score = 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)
    return cosine_sim_score


def get_most_similar(word2vec: KeyedVectors, word : str, k : int) -> list[(str, float)]:
    if word not in word2vec.key_to_index:
        return []
    # Use gensim's most_similar method as its much faster than calling get_cosine_similarity()
    similar_words = word2vec.most_similar(word, topn=k)
    return similar_words
    

def word_analogy(word2vec: KeyedVectors, word1: str, word2: str, word3: str) -> tuple[str, float]:
    # Check if all words exist in the model's vocabulary
    if not all(word in word2vec.key_to_index for word in [word1, word2, word3]):
        return ("", 0.0)
    
    # Doesn't make a call to get_most_similar() as its faster to use gensim's most_similar method
    result = word2vec.most_similar(positive=[word2, word3], negative=[word1], topn=1)
    word4 = result[0]  # Returns tuple of (word, similarity)
    return word4


def cos_sim(A: np.ndarray, B: np.ndarray) -> float:
    dot_product = np.dot(A, B)
    norm1 = np.linalg.norm(A)
    norm2 = np.linalg.norm(B)
    cosine_similarity = 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)

    return cosine_similarity


def get_cos_sim_different_models(word: str, model1: Word2Vec, model2: Word2Vec, cos_sim_function: Callable[[np.ndarray, np.ndarray], float]) -> float:
    try:
        vec1 = model1.wv[word]
        vec2 = model2.wv[word]
    except KeyError:
        return 0.0
    
    cosine_similarity_of_embeddings = cos_sim_function(vec1, vec2)

    return cosine_similarity_of_embeddings

def get_average_cos_sim(word: str, neighbors: list[str], model: Word2Vec, cos_sim_function: Callable[[np.ndarray, np.ndarray], float]) -> float:
    word_vector = model.wv[word]
    similarities = []
    
    for neighbor in neighbors:
        try:
            neighbor_vector = model.wv[neighbor]
            sim = cos_sim_function(word_vector, neighbor_vector)
            similarities.append(sim)
        except KeyError:
            continue
            
    avg_cosine_similarity = np.mean(similarities) if similarities else 0.0
    return avg_cosine_similarity


def create_tfidf_matrix(documents: list[TaggedCorpusView], stopwords: list[str]) -> tuple[np.ndarray, list[str]]:
    # Preprocessing documents
    processed_docs = [
        [word.lower() for word in doc if word.isalnum() and word.lower() not in stopwords] 
        for doc in documents
    ]
    # Create sorted vocabulary and index mapping
    vocabulary = sorted({word for doc in processed_docs for word in doc})
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    num_docs, num_words = len(documents), len(vocabulary)
    
    # Initialize term_frequency_matrix and document frequencies
    tf_matrix = np.zeros((num_docs, num_words))
    doc_freq = np.zeros(num_words)
    
    for i, doc in enumerate(processed_docs):
        word_counts = Counter(doc)
        for word, count in word_counts.items():
            j = vocab_index[word]
            tf_matrix[i, j] = count
            doc_freq[j] += 1
    
    # Compute smoothened IDF using formula log10(N / (df + 1)) + 1
    idf = np.log10(num_docs / (doc_freq + 1)) + 1
    tfidf_matrix = tf_matrix * idf
    
    return tfidf_matrix, vocabulary


def get_idf_values(documents : list[TaggedCorpusView], stopwords: list[str]) -> np.ndarray:
    processed_docs = [
        [word.lower() for word in doc if word.isalnum() and word.lower() not in stopwords] 
        for doc in documents
    ]
    vocabulary = sorted({word for doc in processed_docs for word in doc})
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    doc_freq = np.zeros(len(vocabulary))
    
    for doc in processed_docs:
        word_counts = Counter(doc)
        for word, _ in word_counts.items():
            j = vocab_index[word]
            doc_freq[j] += 1
    
    idf = np.log10(len(documents) / (doc_freq + 1)) + 1
    return idf


def calculate_sparsity(tfidf_matrix: np.ndarray) -> float:
    sparsity = (tfidf_matrix == 0).sum() / tfidf_matrix.size
    return sparsity


def extract_salient_words(VT: np.ndarray, vocabulary: list[str], k: int) -> dict[int, list[str]]:
    n = VT.shape[0]
    salient_words = {}
    
    for dim in range(n):        
        # Sort the weights of each latent dimension
        sorted_indices = np.argsort(VT[dim, :])
        top_k_indices = sorted_indices[-k:]
        
        # Convert indices to words
        salient_words[dim] = [vocabulary[idx] for idx in top_k_indices]
    
    return salient_words


def get_similar_documents(U: np.ndarray, Sigma: np.ndarray, VT: np.ndarray, doc_index: int, k: int) -> list[int]:
    doc_embeddings = U * Sigma
    query_embedding = doc_embeddings[doc_index]
    
    # Calculate cosine similarity between query document and all documents
    similarities = []
    for i in range(len(doc_embeddings)):
        if i != doc_index:  # Exclude the query document
            sim = cos_sim(query_embedding, doc_embeddings[i])
            similarities.append((i, sim))
    
    # Sort by similarity in descending order and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_doc_indices = [idx for idx, _ in similarities[:k]]
    
    return similar_doc_indices


def document_retrieval(vocabulary: list[str], idf_values: np.ndarray, U: np.ndarray, Sigma: np.ndarray, VT: np.ndarray, query: list[str], k: int) -> list[int]:
    # Create vocabulary index mapping and process query
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    query_counts = Counter(w.lower() for w in query if w.isalnum())
    
    # Construct query TF-IDF vector (vocabulary size)
    q_tfidf = np.zeros(len(vocabulary))
    for word, count in query_counts.items():
        if word in vocab_index:
            q_tfidf[vocab_index[word]] = count * idf_values[vocab_index[word]]
    
    # Project query into 10-dimensional LSA space
    # q_tfidf (1 × vocab_size) @ VT.T (vocab_size × 10) = (1 × 10)
    q_lsa = q_tfidf @ VT.T
    
    # Scale query by singular values
    q_embedding = q_lsa / Sigma
    
    # Get document vectors in LSA space
    # U (n_docs × 10) already represents documents in LSA space
    doc_embeddings = U * Sigma  # Broadcasting Sigma across columns
    
    # Compute cosine similarities
    similarities = np.array([cos_sim(q_embedding, doc) for doc in doc_embeddings])
    
    # Return indices of top k most similar documents
    return similarities.argsort()[-k:].tolist()


if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt', "r", encoding='utf-8') as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('data/stop_words.txt', "r", encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    
    ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    print(sample_output.most_common(10))
    """
    [(('the', 'pandemic'), 19811),
    (('a', 'pandemic'), 16615),
    (('a', 'mask'), 14353),
    (('a', 'wear'), 11017),
    (('wear', 'mask'), 10628),
    (('mask', 'wear'), 10628),
    (('do', 'n’t'), 10237),
    (('during', 'pandemic'), 8127),
    (('the', 'covid'), 7630),
    (('to', 'go'), 7527)]
    """
    ### END OF REFERENCE OUTPUT
    
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)
    print(context_counter)

    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817


    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    """Word2Vec for Meaning Change"""

    # Comparing 40-60 year olds in the 1910s and 40-60 year olds in the 2000s
    model_t1 = Word2Vec.load('data/1910s_50yos.model')
    model_t2 = Word2Vec.load('data/2000s_50yos.model')

    # Cosine similarity function for vector inputs
    vector_1 = np.array([1,2,3,4])
    vector_2 = np.array([3,5,4,2])
    cos_similarity = cos_sim(vector_1, vector_2)
    print(cos_similarity)
    # 0.8198915917499229

    # Similarity between embeddings of the same word from different times
    major_cos_similarity = get_cos_sim_different_models("major", model_t1, model_t2, cos_sim)
    print(major_cos_similarity)
    # 0.19302374124526978

    # Average cosine similarity to neighborhood of words
    neighbors_old = ['brigadier', 'colonel', 'lieutenant', 'brevet', 'outrank']
    neighbors_new = ['significant', 'key', 'big', 'biggest', 'huge']
    print(get_average_cos_sim("major", neighbors_old, model_t1, cos_sim))
    # 0.6957747220993042
    print(get_average_cos_sim("major", neighbors_new, model_t1, cos_sim))
    # 0.27042335271835327
    print(get_average_cos_sim("major", neighbors_old, model_t2, cos_sim))
    # 0.2626224756240845
    print(get_average_cos_sim("major", neighbors_new, model_t2, cos_sim))
    # 0.6279034614562988

    ### The takeaway -- When comparing word embeddings from 40-60 year olds in the 1910s and 2000s,
    ###                 (i) cosine similarity to the neighborhood of words related to military ranks goes down;
    ###                 (ii) cosine similarity to the neighborhood of words related to significance goes up.


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # type: ignore # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
