import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq

class EmbeddingFinder:

    def __init__(self, arr_prod, k=10, max_ids=10):

        #self.arr = arr  # array containing the  text and embeddings
        self.k = k #Number of top similar entries to return.
        self.arr_prod = arr_prod # array containing the embeddings and product_ids
        self.max_ids = max_ids # The maximum number of unique product_ids to retrieve.

        self.search_table = pd.DataFrame(self.arr_prod, columns=['embedding', 'product_ids'])
        self.embeddings_df = np.vstack(
            self.search_table['embedding'].values)  # Assuming 'embedding' is a list of embeddings in df

    def find_top_k_similar_large(self, predicted_embedding, k=10):
        """
        Find the top K most similar text and embeddings to the predicted embedding in a large dataset.

        Parameters:
        - arr (array) : array containing the text and embeddings
        - predicted_embedding (np.ndarray): The embedding to compare against.
        - k (int): Number of top similar entries to return.

        Returns:
        - list of dict: A list of dictionaries with 'text', 'embedding', and 'similarity' for the top K entries.
        """
        df = pd.DataFrame(self.arr_prod, columns=['text', 'embedding'])
        predicted_embedding = np.array(predicted_embedding).reshape(1, -1)  # Reshape for compatibility
        top_k = []  # Min-heap to store top K entries

        for _, row in df.iterrows():
            text, embedding = row['text'], np.array(row['embedding'])
            similarity = cosine_similarity(embedding.reshape(1, -1), predicted_embedding).item()

            # Maintain a heap of size k for top K similarities
            if len(top_k) < k:
                heapq.heappush(top_k, (similarity, text, embedding))
            else:
                heapq.heappushpop(top_k, (similarity, text, embedding))

        # Sort by similarity in descending order
        top_k = sorted(top_k, key=lambda x: x[0], reverse=True)
        return [(t, e) for t, e in top_k]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def get_product_ids_for_embeddings(self, predicted_embedding):
        """
        For each text-embedding pair in the dictionary, search the dataframe for the most similar text
        and retrieve corresponding product_ids, ensuring no repetitions. The process continues until
        the list contains 'max_ids' unique product_ids.

        Parameters:
        - arr (array) : array containing the  text and embeddings
        - arr_prod (array) : array containing the embeddings and product_ids
        - max_ids (int): The maximum number of unique product_ids to retrieve.

        Returns:
        - list: A list of unique product_ids based on cosine similarity.
        """

        text_embedding_list = self.find_top_k_similar_large(predicted_embedding)
        unique_product_ids = set()  # To store unique product_ids


        # Iterate over each text-embedding tuple in the list
        for i in range(len(text_embedding_list)):
            for text, embedding in text_embedding_list[i]:

                if len(unique_product_ids) >= self.max_ids:
                    break  # Stop if we've already collected 'max_ids' product_ids

                # Compute cosine similarities between the current text embedding and all embeddings in the DataFrame
                embedding = np.array(embedding).reshape(1, -1)  # Reshape for compatibility
                similarities = cosine_similarity(self.embeddings_df, embedding).flatten()

                # Get the indices of the most similar entries
                similar_indices = np.argsort(similarities)[::-1]

                # Iterate through the sorted indices and collect unique product_ids
                for idx in similar_indices:
                    if len(unique_product_ids) >= self.max_ids:
                        break  # Stop once we reach the limit

                    # Extract and split product IDs (assumes product_ids is a comma-separated string)
                    product_ids = self.search_table.iloc[idx]['product_ids'].split(',')
                    unique_product_ids.update(product_ids)  # Add multiple IDs to the set at once

        # Create a ranked list of product IDs
        ranked_product_ids = [(product_id, rank + 1) for rank, product_id in enumerate(unique_product_ids)]

        return ranked_product_ids