import spacy
import streamlit as st
from spacy import displacy
from process_new_data import get_mwe_predictions

attributes = {
    'IDIOM': {
        'definition': 'A phrase that could have a literal meaning but does not in the given context',
        'color': '#D077A3',
        'full_name': 'Idiom'},
    'V-P_CONSTRUCTION': {
        'definition': 'A verb followed by a particle, potentially with an intervening constituent, '
                      'that act as a single verb',
        'color': '#FF7878',
        'full_name': 'Verb-particle construction'},
    'NN_COMP': {
        'definition': 'A sequence of two or more nouns that behave syntactically as a single noun',
        'color': '#FD9B4D',
        'full_name': 'Noun-noun compound'},
    'LIGHT_V': {
        'definition': 'An expression starting with a verb that has little semantic content followed by another '
                      'word or phrase that carries the entire expression’s meaning',
        'color': '#FFF1A9',
        'full_name': 'Light verb construction'}}

st.html('<h1>Multiword Expression Extraction</h1>')
intro_tab, mwe_tab, model_tab, demo_tab = st.tabs((r'$\textsf{\large Introduction}$', r'$\textsf{\large About MWES}$',
                                                   r'$\textsf{\large Model details}$', r'$\textsf{\large Demonstration}$'))

with intro_tab:
    st.html('<h2>Goal and Motivation</h2>')
    st.html('<h2>Approaches</h2>')
    st.html('<h2>Results</h2>')

with mwe_tab:
    st.write('A multiword expression is a string of words which act as a single semantic or syntactic element. '
             'For this project, I extract four types of expressions:')
    for mwe_type in attributes:
        st.html(f'<li><mark class="entity" style="background: {attributes[mwe_type]["color"]}; padding: 0.45em 0.6em; '
                f'margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">{attributes[mwe_type]["full_name"]} '
                f'({mwe_type})</mark>: {attributes[mwe_type]["definition"]}.</li>')

with model_tab:
    st.write('Technical details')
    st.table({'MWE Type': ['NN_COMP', 'V-P_CONSTRUCTION', 'LIGHT_V', 'IDIOM'],
              'Model': ['BERT-CRF', 'BERT-CRF', 'BERT-CRF', 'BERT-CRF'],
              'Trained on': ['Noun-noun compounds', 'All MWEs', 'All MWEs', 'All MWEs'],
              'Learning Rate': [0.0001, 0.0001, 0.0001, 0.0001],
              'Batch Size': [4, 16, 16, 8],
              'Number of Epochs': [5, 10, 10, 10],
              'F1 on Dev Set': [.7624, .4857, .3571, .1944]})

with demo_tab:
    user_input = st.text_area('Input text from which to extract multiword expressions below', height=100)

    if len(user_input) > 0:
        nlp = spacy.blank('en')
        doc = nlp.make_doc(user_input)
        mwe_spans = get_mwe_predictions(user_input)

        mwes = []
        for label, start, end in mwe_spans:
            mwe = doc.char_span(start, end, label=label, alignment_mode='contract')
            if mwe is not None:
                mwes.append(mwe)
        doc.set_ents(mwes)

        if len(mwes) == 0:
            st.write('No multiword expressions found.')
        else:
            options = {"ents": ['NN_COMP', 'LIGHT_V', 'V-P_CONSTRUCTION', 'IDIOM'],
                       "colors": {mwe_type: attributes[mwe_type]['color'] for mwe_type in attributes}}
            st.html(displacy.render(doc, style='ent', options=options))
