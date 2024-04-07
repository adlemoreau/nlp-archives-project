import pandas as pd
import yaml
import re


def load_df(file_path):
    """Load a DataFrame from a JSON file and return it."""
    df = pd.read_json(file_path, orient='index', )
    df.rename(columns={df.columns[0]: 'Information'}, inplace=True)
    return df

def load_tags(file_path):
    """Load tags from a YAML file and return them."""
    with open(file_path, 'r') as file:
        tags = yaml.safe_load(file)
    return tags

def create_dict_tags(input_dict):
    """Create a dictionary with the start markers as keys and the categories as values."""
    # Initialize an empty dictionary to hold the tags and their corresponding categories
    tags_categories = {}

    # Iterate over the input dictionary to extract the start markers and their corresponding categories
    for category, markers in input_dict.items():
        # Assign the start marker as the key and the category name as the value in the tags_categories dictionary
        tags_categories[markers['start']] = category

    return tags_categories

def create_list_tags(input_dict):
    """Create a list with the start markers."""
    tags_symbols = []
    for category in input_dict:
        tags_symbols.append(input_dict[category]['start'])
    return tags_symbols

def classify_entry(entry, tags_symbols, tags_categories):
    """Take an entry and classify each part according to the tags"""
    # Split the entry by the tags
    parts = re.split('('+ '|'.join(re.escape(tag) for tag in tags_symbols) +')', entry)

    # Dictionary to hold the classified parts with categories as keys
    classified_parts = {category: None for category in tags_categories.values()}

    # Process the split parts and classify them according to the tags
    for i in range(1, len(parts), 2):  # iterate over every second element (tags) starting from index 1
        tag = parts[i]
        value = parts[i+1].strip()  # get the value after the tag
        category = tags_categories.get(tag, None)  # get the category for the tag
        if category:
            classified_parts[category] = value  # assign the value to the correct category

    return classified_parts

def load_and_create_ground_truth_df(input_df, tags):
    """Load the entities.json file and create a DataFrame with the classified data."""
    tags_dict = create_dict_tags(tags)
    tags_list = create_list_tags(tags)

    classified_data = []
    # Concatenate all the dataframes from the entities.json file
    for i in range(len(input_df)):
        for entry in input_df['Information'].iloc[i].split('\n'):
            classified_data.append(classify_entry(entry, tags_list, tags_dict))

    # Convert the classified data into a DataFrame
    classified_df = pd.DataFrame(classified_data)
    return classified_df