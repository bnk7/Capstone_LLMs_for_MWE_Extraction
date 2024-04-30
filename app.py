import spacy
import streamlit as st
from spacy import displacy
from process_new_data import get_mwe_predictions

nlp = spacy.blank('en')
example = 'This is a test of a noun noun compound. Go figure!'

user_input = st.text_area('Text to process', height=100)

doc = nlp.make_doc(user_input)
mwe_spans = get_mwe_predictions(user_input)
# mwe_spans = [('NN_COMP', 0, 7), ('LIGHT_V', 10, 19)]

mwes = []
for label, start, end in mwe_spans:
    mwe = doc.char_span(start, end, label=label, alignment_mode='contract')
    if mwe is not None:
        mwes.append(mwe)
doc.set_ents(mwes)

colors = {'NN_COMP': '#FD9B4D', 'LIGHT_V': '#FFF1A9', 'V-P_CONSTRUCTION': '#FF7878', 'IDIOM': '#D077A3'}
options = {"ents": ['NN_COMP', 'LIGHT_V', 'V-P_CONSTRUCTION', 'IDIOM'], "colors": colors}
st.html(displacy.render(doc, style='ent', options=options))
