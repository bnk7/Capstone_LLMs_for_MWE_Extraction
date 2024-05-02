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
intro_tab, mwe_tab, model_tab, demo_tab, ref_tab = st.tabs((r'$\textsf{\large Introduction}$',
                                                            r'$\textsf{\large About MWES}$',
                                                            r'$\textsf{\large Model details}$',
                                                            r'$\textsf{\large Demonstration}$',
                                                            r'$\textsf{\large References}$'))

with intro_tab:
    st.html('<h2>Goal and Motivation</h2>')
    st.write('The goal of this project is to train a model on a custom dataset to extract multiword expressions using '
             'encoder and decoder large language models. The data come from the free sample of the Corpus of '
             'Contemporary American English (Davies, 2008-) with annotation by Kilcline et al. (2023). Multiword '
             'expressions (MWEs) are phrases that behave as a single semantic or syntactic unit. Successfully '
             'identifying them can improve the performance of downstream tasks such as parsing and machine '
             'translation (Constant et al., 2017).')
    st.html('<h2>Approaches</h2>')
    st.html("<b>GPT 3.5:</b> I used few-shot learning on GPT 3.5 Turbo with chain-of-thought prompting, following "
            "Shen et al.’s (2023) named entity recognition prompt template.")
    st.html("<b>BERT:</b> I built two models with uncased base BERT (Devlin et al., 2019) at the core. The first has "
            "a simple linear layer and softmax on top of the BERT output. The second uses a CRF.")
    st.html("<b>Ensemble:</b> I also trained models to predict a single type of multiword expression. I chose the "
            "highest-performing model and specification for each MWE type, regardless of whether it was trained on "
            "all MWEs or just one type. I then combined the model predictions.")
    st.html('<h2>Results</h2>')
    st.write("I improved the performance over the GPT 3.5 baseline by almost 30 F1 points using ensemble learning. "
             "For more information on the best performing model, see the model details tab. To see it in use, "
             "navigate to the demonstration.")
    st.html("""<table style="width:50%">
  <tr>
    <th>Model</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
  </tr>
  <tr>
    <td>GPT 3.5</td>
    <td>16.67</td>
    <td>16.20</td>
    <td>16.43</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>36.78</td>
    <td>44.14</td>
    <td>40.13</td>
  </tr>
  <tr>
    <td>BERT-CRF</td>
    <td>38.92</td>
    <td>49.66</td>
    <td>43.64</td>
  </tr>
  <tr>
    <td>Ensemble</td>
    <td>40.31</td>
    <td>54.48</td>
    <td><b>46.33</b></td>
  </tr>
</table>""")

with mwe_tab:
    st.write('A multiword expression is a string of words which act as a single semantic or syntactic element. '
             'I focus on four types of expressions in this project:')
    for mwe_type in attributes:
        st.html(f'<li><mark class="entity" style="background: {attributes[mwe_type]["color"]}; padding: 0.45em 0.6em; '
                f'margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">{attributes[mwe_type]["full_name"]} '
                f'({mwe_type})</mark>: {attributes[mwe_type]["definition"]}.</li>')

with model_tab:
    st.write('The highest performing model was the ensemble model with an F1 of 46.33. Once I determined the best '
             'model for each type of multiword expression, I combined their predictions in order of their '
             'reliability, beginning with the most accurate MWE type and iteratively adding in non-conflicting '
             'predictions for other MWE types.')
    st.html("""<table style="width:100%">
      <tr>
        <th>MWE Type</th>
        <th>Model</th>
        <th style="width:15%">Trained on</th>
        <th style="width:10%">Learning Rate</th>
        <th style="width:7%">Batch Size</th>
        <th style="width:12%">Number of Epochs</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
      <tr>
        <td>NN_COMP</td>
        <td>BERT-CRF</td>
        <td>Noun-noun compounds</td>
        <td>0.0001</td>
        <td>4</td>
        <td>5</td>
        <td>50.94</td>
        <td>80.60</td>
        <td>62.43</td>
      </tr>
      <tr>
        <td>V-P_CONST</td>
        <td>BERT-CRF</td>
        <td>All MWEs</td>
        <td>0.0001</td>
        <td>16</td>
        <td>10</td>
        <td>37.14</td>
        <td>44.83</td>
        <td>40.63</td>
      </tr>
      <tr>
        <td>LIGHT_V</td>
        <td>BERT-CRF</td>
        <td>All MWEs</td>
        <td>0.0001</td>
        <td>16</td>
        <td>10</td>
        <td>41.18</td>
        <td>63.64</td>
        <td>50.00</td>
      </tr>
      <tr>
        <td>IDIOM</td>
        <td>BERT-CRF</td>
        <td>All MWEs</td>
        <td>0.0001</td>
        <td>8</td>
        <td>10</td>
        <td>13.16</td>
        <td>13.16</td>
        <td>13.16</td>
      </tr>
      <tr>
        <td>Overall</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>40.31</td>
        <td>54.48</td>
        <td>46.33</td>
      </tr>
    </table>""")
    st.write('The BERT-CRF model, which uses Viterbi decoding, performed the best for each MWE type. Additionally, '
             'my experiments showed that training on only one type of multiword expression was beneficial for '
             'noun-noun compounds, the most frequent type of MWE in the data, but detrimental for the other, '
             'less common types.')

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

with ref_tab:
    st.html('Mathieu Constant, Gülsen Eryigit, Johanna Monti, Lonneke van der Plas, Carlos Ramisch, Michael Rosner, '
            'and Amalia Todirascu. 2017. <a href="https://direct.mit.edu/coli/article/43/4/837/1581/Multiword'
            '-Expression-Processing-A-Survey">Survey: Multiword expression processing: A Survey</a>. <i>Computational '
            'Linguistics</i>, 43(4):837–892.')
    st.html('Mark Davies. 2008-. <a href="https://www.english-corpora.org/coca/">Corpus of Contemporary American '
            'English (COCA)</a>.')
    st.html('Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. <a '
            'href="https://aclanthology.org/N19-1423/">BERT: Pre-training of deep bidirectional transformers for '
            'language understanding</a>. In <i>Proceedings of the 2019 Conference of the '
            'North American Chapter of the Association for Computational Linguistics: Human Language Technologies, '
            'Volume 1 (Long and Short Papers)</i>, pages 4171–4186, Minneapolis, Minnesota. Association for '
            'Computational Linguistics.')
    st.html('Brynna Kilcline, Gabby Masini, Ruth Rosenblum, and Annika Sparrell. 2023. <a '
            'href="https://github.com/bnk7/Annotation_Project_on_Multiword_Expressions">Multiword expressions</a>.')
    st.html('Yongliang Shen, Zeqi Tan, Shuhui Wu, Wenqi Zhang, Rongsheng Zhang, Yadong Xi, Weiming Lu, and Yueting '
            'Zhuang. 2023. <a href="https://aclanthology.org/2023.acl-long.698/">PromptNER: Prompt locating and '
            'typing for named entity recognition</a>. In <i>Proceedings of the 61st Annual Meeting of the Association '
            'for Computational Linguistics (Volume 1: Long Papers)</i>, '
            'pages 12492–12507, Toronto, Canada. Association for Computational Linguistics.')
