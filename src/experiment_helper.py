import pandas as pd
import random
import numpy as np
from datasets import Dataset, DatasetDict
from datasets import ClassLabel, Sequence
from IPython.display import display, HTML


def extract_text_tags_with_id_from_df(df, category_to_index):
    """Extracts the text and tags from the DataFrame and returns them as lists. 
    The letter that prefixes each ner_tag indicates the token position of the entity:
        - B- indicates the beginning of an entity.
        - I- indicates a token is contained inside the same entity (for example, the State token is a part of an entity like Empire State Building).

    Args:
        df (pd.DataFrame): The DataFrame containing the text and tags.
        category_to_index (dict): A dictionary mapping the categories to their corresponding indices.

    Returns:
        output (list): A list of dictionaries containing the text, tags, and ID for each row.
    """
    output = []

    for index, row in df.iterrows():
        words_list = []
        categories_list = []

        for column in df.columns:
            value = row[column]
            if pd.notnull(value) and isinstance(value, str):
                split_values = value.split()
                for i, item in enumerate(split_values):
                    # Determine the tag (B- or I-)
                    tag = f'B-{column}' if i == 0 else f'I-{column}'

                    # Append the word and its category index
                    words_list.append(item)
                    # If the tag is not found in the predefined mapping, append 0 (for 'O')
                    categories_list.append(category_to_index.get(tag, 0))
            elif pd.notnull(value):
                words_list.append(str(value))
                categories_list.append(0)  # Non-categorical or non-string values

        # For any remaining space, mark as 'O'
        categories_list += [0] * (len(words_list) - len(categories_list))

        # Append the dictionary for this row to the output list
        output_dict = {'id': str(index), 'ner_tags': categories_list, 'tokens': words_list}
        output.append(output_dict)

    return output

def split_dataset(texts_tags, train_ratio=0.7, test_ratio=0.15, validation_ratio=0.15):
    """
    Shuffle and split the dataset into training, validation, and test sets.

    Parameters:
    texts_tags (list): The dataset to split.
    train_ratio (float): The proportion of the dataset to include in the train split.
    test_ratio (float): The proportion of the dataset to include in the test split.
    validation_ratio (float): The proportion of the dataset to include in the validation split.

    Returns:
    tuple: A tuple containing three lists: (train_data, validation_data, test_data).
    """

    # Shuffle the dataset to ensure random distribution
    random.shuffle(texts_tags)

    # Calculate the split indices
    split_index_1 = int(len(texts_tags) * train_ratio)
    split_index_2 = int(len(texts_tags) * (train_ratio + test_ratio))

    # Split the data into training, validation, and test sets
    train_data = texts_tags[:split_index_1]
    validation_data = texts_tags[split_index_1:split_index_2]
    test_data = texts_tags[split_index_2:]

    return train_data, validation_data, test_data

class MyDataset:
    def __init__(self, train_data, validation_data, test_data):
        # Convert list of dictionaries to Dataset directly without additional formatting
        self.dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(pd.DataFrame(train_data)),
            'validation': Dataset.from_pandas(pd.DataFrame(validation_data)),
            'test': Dataset.from_pandas(pd.DataFrame(test_data)),
        })

    def get_dataset(self):
        return self.dataset_dict
    
def show_random_elements(dataset, tokenizer, num_examples=10, ):
    """Display `num_examples` of random elements from the dataset with their features."""
    assert num_examples <= len(dataset)
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
            
    # add tokenization column
    df['tokenized'] = df['tokens']
    df['tokenized'] = df['tokenized'].apply(lambda x: tokenizer(x, is_split_into_words=True))
    df['tokenized'] = df['tokenized'].apply(lambda x: tokenizer.convert_ids_to_tokens(x["input_ids"]))
      
    display(HTML(df.to_html()))
    
def tokenize_and_align_labels(examples, tokenizer):
    """Réaligne les tokens et les étiquettes, et tronquer les séquences pour qu'elles ne soient pas plus longues que la longueur d'entrée maximale

    Args:
        examples (dict): Une entrée de l'ensemble de données.

    Returns:
        tokenized_inputs (dict): Les tokens encodés et alignés avec les étiquettes.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=False, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx: 
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs