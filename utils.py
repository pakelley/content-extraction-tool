import os
import re
import logging
import lxml
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import ipywidgets as widgets
from IPython.display import HTML, display
from traitlets import Unicode

from dragnet import Extractor
from dragnet.blocks import simple_tokenizer as tokenizer
from dragnet.compat import train_test_split
from dragnet.data_processing import prepare_all_data, extract_all_gold_standard_data, get_filenames
from dragnet.util import evaluation_metrics
import justext

log = logging.getLogger()
log.setLevel('ERROR')

def init_cleaner():
    cleaner = lxml.html.clean.Cleaner()

    cleaner.add_nofollow = True
    # cleaner.scripts = False
    # cleaner.javascript = False
    cleaner.comments = False
    cleaner.style = False
    cleaner.inline_style = False
    cleaner.embedded = False
    cleaner.forms = False
    cleaner.frames = False
    cleaner.annoying_tags = False
    cleaner.links = False
    cleaner.meta = False
    cleaner.page_structure = False
    cleaner.processing_instructions = False
    cleaner.remove_unknown_tags = False
    cleaner.safe_attrs_only = False

    return cleaner

def read_dragnet_data(data_dir, to_extract='content'):
    """
    Given a directory with `dragnet`-formatted data, return
    a `pandas.DataFrame` for both test and train data, both containing
    'fileroot', 'doc', 'labels', 'weights', and 'tokens' for each document
    """
    # read data
    urls_df = pd.read_csv(os.path.join(data_dir, 'URLs_summary.csv'))
    urls_df['fileroot'] = urls_df['html_file_name'].apply(lambda fname: re.sub(r'\.html', '', fname))
    extract_all_gold_standard_data(data_dir)
    data = prepare_all_data(data_dir)
    # parse data
    data_df = pd.DataFrame({'corrected_paths': list(get_filenames(os.path.join(data_dir, 'Corrected')))})
    data_df['fileroot'] = data_df['corrected_paths'].apply(lambda fname: re.sub(r'\.html.corrected.txt', '', fname))
    data_df.drop(columns='corrected_paths')
    
    data_df['doc'] = [item[0] for item in data]
    tmp_extractor = Extractor(to_extract=to_extract)
    labels_weights = [tmp_extractor._get_labels_and_weights(item[1], item[2]) for item in data]
    data_df['labels'] = [item[0] for item in labels_weights]
    data_df['weights'] = [item[1] for item in labels_weights]
    data_df['tokens'] = get_gs_tokens(data, to_extract)
    
    data_df = data_df.join(urls_df.set_index('fileroot'), on='fileroot').rename(columns={'URL': 'url'})

    return data_df

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
    base_content = [base_ext.extract(x) for x in tqdm(df['doc'], desc='Extracting Dragnet Content', leave=False)]
    comp_content = [comp_ext.extract(x) for x in tqdm(df['doc'], desc='Extracting Justext Content', leave=False)]
    expected_content = [get_gs_content(base_ext, x, label) for x, label in 
                        tqdm(zip(df['doc'], df['labels']), total=len(df['doc']), desc='Parsing Expected Content', leave=False)]

    base_tokens = [tokenizer(c) for c in base_content]
    comp_tokens = [tokenizer(c) for c in comp_content]
    expected_tokens = [tokenizer(c) for c in expected_content]

    base_scores = [score_dragnet(predicted, expected, weight)
                   for predicted, expected, weight in zip(base_tokens, expected_tokens, df['weights'])]
    comp_scores = [score_dragnet(predicted, expected, weight)
                   for predicted, expected, weight in zip(comp_tokens, expected_tokens, df['weights'])]

    return pd.DataFrame({
        'fileroot': df['fileroot'],
        'url': df['url'],
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


class FancySlider(widgets.HBox):
    def __init__(self, max=20, value=0):
        _view_name = Unicode('FancySlider').tag(sync=True)
        
        self.slider = widgets.IntSlider(value=value, min=0, max=max, 
            continuous_update=False, readout=False, step=1,
            layout=widgets.Layout(width='15em'))
        
        button_layout = widgets.Layout(width='2.5em', height='2.5em')
        self.plus_button = widgets.Button(icon='plus', button_style='', layout=button_layout)
        self.minus_button = widgets.Button(icon='minus', button_style='', layout=button_layout)
        
        self.plus_button.on_click(self.increment_slider)
        self.minus_button.on_click(self.decrement_slider)

        self.textbox = widgets.Text(str(self.value), layout=widgets.Layout(width='2.2em'))
        self.textbox.observe(self.update_slider, 'value')
        self.slider.observe(self.update_textbox, 'value')
        
        super(FancySlider, self).__init__([self.minus_button, self.plus_button, self.slider, self.textbox])

    @property
    def value(self):
        return self.slider.value
    
    @value.setter
    def value(self, val):
        self.slider.value = val

    def increment_slider(self, *args):
        self.value += 1
        self.update_textbox()

    def decrement_slider(self, *args):
        self.value -= 1
        self.update_textbox()

    def update_slider(self, *args):
        if self.textbox.value is not '':
            self.value = int(self.textbox.value)

    def update_textbox(self, *args):
        self.textbox.value = str(self.value)
        
    def observe(self, *args, **kwargs):
        if 'slider' in dir(self):
            self.slider.observe(*args, **kwargs)

class ExtractionAccordion(widgets.Accordion):
    def __init__(self, slider, df):
        # build accordion for content
        self.df = df
        self.expected = widgets.interactive_output(self.print_expected, 
                                                       {'idx': slider})
        self.dragnet = widgets.interactive_output(self.print_dragnet, 
                                                      {'idx': slider})
        self.justext = widgets.interactive_output(self.print_justext, 
                                                      {'idx': slider})
        
        (super(ExtractionAccordion, self).__init__(
            children=[self.expected, self.dragnet, self.justext]))
        
        self.set_title(0, 'Expected')
        self.set_title(1, 'Dragnet')
        self.set_title(2, 'Justext')

        accordion = widgets.Accordion()
        
    def print_expected(self, idx):
        print(self.df['expected_content'].iloc[idx])

    def print_dragnet(self, idx):
        print('f1: ', self.df['base_f1'].iloc[idx])
        print(self.df['base_content'].iloc[idx])

    def print_justext(self, idx):
        print('f1: ', self.df['comp_f1'].iloc[idx])
        print(self.df['comp_content'].iloc[idx])

class ContentExtractComparisonWidget(widgets.Tab):
    """creates a widget for analyzing content extraction"""
    def __init__(self, df):
        self.slider = FancySlider(max=df.count()['base_content']-1, value=0)
        self.df = df
        
        self.accordion = ExtractionAccordion(self.slider, self.df)
        
        # build HTML views
        self.cleaner = init_cleaner() # TODO
        self.html_view = widgets.interactive_output(self.show_html, {'idx': self.slider})
        self.raw_html_view = widgets.interactive_output(lambda idx: print(df['test_data'].iloc[idx]), {'idx': self.slider})
        self.file_label = widgets.interactive_output(self.get_file_label, {'idx': self.slider})
        self.url_view = widgets.interactive_output(self.show_url, {'idx': self.slider})

        # build error mode labels
        if 'reviewed' not in df.columns:
            df['reviewed'] = pd.Series([False] * len(df.index), index=df.index)

            
        self.reviewed_cbox = widgets.ToggleButton(
            value=bool(df.loc[self.slider.value, 'reviewed']),
            description='Reviewed',
            layout=widgets.Layout(width='8em')
        )
        self.stylize_reviewed_cbox()
            
        self.reviewed_cbox.observe(self.update_reviewed_df, 'value')
        self.reviewed_cbox.observe(self.stylize_reviewed_cbox, 'value')
        self.slider.observe(self.update_reviewed_cbox, 'value')

        # make content entry field
        self.content_text = widgets.Textarea(df['labeled_content'].iloc[self.slider.value],
                                        layout=widgets.Layout(width='auto'))


        self.content_text.observe(self.update_content, 'value')
        self.slider.observe(self.update_content_text, 'value')

        # collect into tabs
        (super(ContentExtractComparisonWidget, self).__init__([self.accordion, self.html_view, self.raw_html_view, self.content_text]))
        self.set_title(0, 'Extracted Content')
        self.set_title(1, 'Rendered HTML')
        self.set_title(2, 'Raw HTML')
        self.set_title(3, 'Label Content')
    
    def show_html(self, idx):
        html_str = self.df.loc[idx, 'test_data']
        clean_html_str = self.cleaner.clean_html(html_str)
        display(HTML(clean_html_str))
        
    def show_url(self, idx):
        display(HTML("<a href=\"{url}\">{url}</a>".format(url=self.df.loc[idx, 'url'])))
        
    def get_file_label(self, idx):
        display(widgets.Label('Fileroot:  {fileroot}'.format(fileroot=self.df.loc[idx, 'fileroot'])))
    
    def update_reviewed_df(self, *args):
        self.df.loc[self.slider.value, 'reviewed'] = self.reviewed_cbox.value
        
    def update_reviewed_cbox(self, *args):
        self.reviewed_cbox.value = bool(self.df.loc[self.slider.value, 'reviewed'])
        self.stylize_reviewed_cbox()
        
    def stylize_reviewed_cbox(self, *args):
        self.reviewed_cbox.icon = 'check' if self.reviewed_cbox.value else 'times'
        self.reviewed_cbox.button_style = 'success' if self.reviewed_cbox.value else ''
        
    def update_content(self, *args):
        self.df.loc[self.slider.value, 'labeled_content'] = self.content_text.value
        
    def update_content_text(self, *args):
        self.content_text.value = self.df.loc[self.slider.value, 'labeled_content']
        
    def _ipython_display_(self, *args):
        composed_widget = widgets.VBox([
            widgets.HBox([self.slider, self.reviewed_cbox]),
            widgets.HBox([widgets.HBox([widgets.Label('URL:  '), self.url_view]),
                          self.file_label]),
            widgets.Label("Content/comments separator: !@#$%^&*() COMMENTS"), 
            self])
        return composed_widget._ipython_display_(*args)


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
