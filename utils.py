import os
import re
import pandas as pd
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import HTML, display

from dragnet import Extractor
from dragnet.blocks import simple_tokenizer as tokenizer
from dragnet.compat import train_test_split
from dragnet.data_processing import prepare_all_data, extract_all_gold_standard_data, get_filenames
from dragnet.util import evaluation_metrics
import justext

def read_dragnet_data(data_dir, to_extract='content'):
    """
    Given a directory with `dragnet`-formatted data, return
    a `pandas.DataFrame` for both test and train data, both containing
    'fileroot', 'doc', 'labels', 'weights', and 'tokens' for each document
    """
    # read data
    filenames = [re.sub(r'\.html', '', fname)
                 for fname in list(get_filenames(os.path.join(data_dir, 'HTML')))]
    extract_all_gold_standard_data(data_dir)
    # split data
    data = prepare_all_data(data_dir)
    train_data_fnames, test_data_fnames = train_test_split(list(zip(data, filenames)))
    del data
    # parse data
    test_df = make_dragnet_df(test_data_fnames, 'content')
    del test_data_fnames
    train_df = make_dragnet_df(train_data_fnames, 'content')
    del train_data_fnames

    return train_df, test_df

def make_dragnet_df(data_and_filenames, to_extract):
    data = [item[0] for item in data_and_filenames]
    filenames = [item[1] for item in data_and_filenames]
    tmp_extractor = Extractor(to_extract=to_extract)
    del data_and_filenames
    docs = [item[0] for item in data]
    labels_weights = [tmp_extractor._get_labels_and_weights(item[1], item[2]) for item in data]
    labels = [item[0] for item in labels_weights]
    weights = [item[1] for item in labels_weights]
    tokens = get_gs_tokens(data, tmp_extractor.to_extract)
    del data
    del labels_weights

    return pd.DataFrame({
        'filename': filenames,
        'doc': docs,
        'labels': labels,
        'weights': weights,
        'tokens': tokens
    })

def get_gs_content(extractor, doc, labels):
    """util to get labeled content from a doc"""
    blocks = extractor.blockifier.blockify(doc)
    labeled_blocks = [block for block, lbl in zip(blocks, labels) if lbl == 1]
    return '\n'.join(block.text for block in labeled_blocks)

def get_gs_tokens(dataset, to_extract):
    """util to parse tokens from a dataset"""
    if ('content' in to_extract) and ('comments' in to_extract):
        return [d[1][2] + d[2][2] for d in dataset]
    elif 'content' in to_extract:
        return [d[1][2] for d in dataset]
    elif 'comments' in to_extract:
        return [d[2][2] for d in dataset]
    else:
        print('Invalid value for `to_extract`')
        return None

def score_dragnet(predicted, expected, weights):
    """util to get score, but return 1's for correctly predicting no content"""
    if len(expected) > 0 and len(predicted) > 0:
        return evaluation_metrics(predicted, expected)
    elif len(expected) == 0 and len(predicted) == 0:
        return (1.0, 1.0, 1.0)
    else:
        return (0.0, 0.0, 0.0)

def extraction_comparison(base_ext, comp_ext, df, npartitions=1):
    """returns a dataframe containing info from each step of extraction"""
    base_content = [base_ext.extract(x) for x in tqdm(df['doc'], desc='Extracting Dragnet Content')]
    comp_content = [comp_ext.extract(x) for x in tqdm(df['doc'], desc='Extracting Justext Content')]
    expected_content = [get_gs_content(base_ext, x, label) for x, label in 
                        tqdm(zip(df['doc'], df['labels']), total=len(df['doc']), desc='Parsing Expected Content')]
    
    base_tokens = [tokenizer(c) for c in base_content]
    comp_tokens = [tokenizer(c) for c in comp_content]
    expected_tokens = [tokenizer(c) for c in expected_content]
    
    base_scores = [score_dragnet(predicted, expected, weight)
                   for predicted, expected, weight in zip(base_tokens, expected_tokens, df['weights'])]
    comp_scores = [score_dragnet(predicted, expected, weight)
                   for predicted, expected, weight in zip(comp_tokens, expected_tokens, df['weights'])]
    
    return pd.DataFrame({
        'fileroot': df['filename'],
        'test_data': df['doc'],
        'test_labels': df['labels'],
        'test_weights': df['weights'],
        'base_content': base_content,
        'comp_content': comp_content,
        'expected_content': expected_content,
        'labeled_content': expected_content,
        'base_tokens': base_tokens,
        'comp_tokens': comp_tokens,
        'expected_tokens': expected_tokens,
        'base_f1': [score[2] for score in base_scores],
        'comp_f1': [score[2] for score in comp_scores]
    })


def content_extract_comparison_widget(df):
    """creates a widget for analyzing content extraction"""
    def print_expected(df, idx):
        print(df['expected_content'].iloc[idx])

    def print_dragnet(df, idx):
        print('f1: ', df['base_f1'].iloc[idx])
        print(df['base_content'].iloc[idx])

    def print_justext(df, idx):
        print('f1: ', df['comp_f1'].iloc[idx])
        print(df['comp_content'].iloc[idx])

    slider = widgets.IntSlider(min=0, max=df.count()['base_content']-1, step=1, value=0, continuous_update=False)

    # build accordion for content
    expected_interact = widgets.interactive_output(print_expected, {'df': widgets.fixed(df), 'idx': slider})
    dragnet_interact = widgets.interactive_output(print_dragnet, {'df': widgets.fixed(df), 'idx': slider})
    justext_interact = widgets.interactive_output(print_justext, {'df': widgets.fixed(df), 'idx': slider})

    accordion = widgets.Accordion(children=[expected_interact, dragnet_interact, justext_interact])
    accordion.set_title(0, 'Expected')
    accordion.set_title(1, 'Dragnet')
    accordion.set_title(2, 'Justext')

    # build HTML view
    def show_html(df, idx):
        display(HTML(df['test_data'].iloc[idx]))
    html_view = widgets.interactive_output(show_html, {'df':widgets.fixed(df), 'idx':slider})

    # build error mode labels
    if 'error_modes' not in df.columns:
        df['error_modes'] = pd.Series([''] * len(df.index), index=df.index)
        
    em_text = widgets.Text(df['error_modes'].iloc[slider.value])
    
    def update_em(*args):
        df['error_modes'].iloc[slider.value] = em_text.value
    em_text.observe(update_em, 'value')
    
    def update_text(*args):
        em_text.value = df['error_modes'].iloc[slider.value]
    slider.observe(update_text, 'value')
    
    # make content entry field
    content_text = widgets.Textarea(df['labeled_content'].iloc[slider.value])
    
    def update_content(*args):
        df['labeled_content'].iloc[slider.value] = content_text.value
    content_text.observe(update_content, 'value')
    
    def update_content_text(*args):
        content_text.value = df['labeled_content'].iloc[slider.value]
    slider.observe(update_content_text, 'value')
    
    # collect into tabs
    tab_nest = widgets.Tab()
    tab_nest.children = [accordion, html_view, content_text]
    tab_nest.set_title(0, 'Extracted Content')
    tab_nest.set_title(1, 'HTML')
    tab_nest.set_title(2, 'Label Content')

    # final display
    return widgets.VBox([widgets.HBox([slider, widgets.Label('Error Mode:'), em_text]), tab_nest])


class JustextWrapper():
    
    def extract(self, html_content):
        stripped_content = re.sub('\s+', ' ', html_content.strip())
        if stripped_content:
            return ' '.join(list(map(
                lambda p: re.sub('\s+', ' ', p.text),
                list(filter(
                    lambda p: not p.is_boilerplate,
                    justext.justext(
                        stripped_content,
                        justext.get_stoplist(
                            "English")))))))
