import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from tqdm import tqdm
import json
import os

# =========== spacy customization =========== #
# Maintain intra-word hyphen
def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

nlp = spacy.load('en_core_web_trf')
nlp.tokenizer = custom_tokenizer(nlp)
# =========================================== #

def modify_sentence(sentence, *pos):
    doc = nlp(sentence)
    tokens = [token for token in doc]
    modified = ""

    if len(pos) == 1:
        pos1 = pos[0]
        for i, token in enumerate(tokens):
            if i == pos1:
                modified += ('<' + token.text + '>')
                modified += token.whitespace_
            else:
                modified += token.text_with_ws
    elif len(pos) == 2:
        pos1, pos2 = pos
        for i, token in enumerate(tokens):
            if i == pos1 or i == pos2:
                modified += ('<' + token.text + '>')
                modified += token.whitespace_
            else:
                modified += token.text_with_ws
    else:
        raise Exception("The number of pos parameter can only be one or two.")
    
    return modified

def generate(file_path):
    result = list()

    with open(file=file_path, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)
    relations = data['relations']
    for relation in relations:
        d = dict()
        sid1 = relation['sid1']
        sid2 = relation['sid2']
        sentence1 = relation['sent1']
        sentence2 = relation['sent2']
        pos1 = relation['pos1']
        pos2 = relation['pos2']
        tlink = relation['tlink']
        new_sentence = ""
        if sid1 == sid2:
            new_sentence = modify_sentence(sentence1, pos1, pos2)
        else:
            new_sentence = modify_sentence(sentence1, pos1)
            new_sentence = new_sentence + ' '
            new_sentence = new_sentence + modify_sentence(sentence2, pos2)
        prompt = new_sentence
        d['prompt'] = prompt
        d['answer'] = tlink
        result.append(d)
    return result

if __name__ == '__main__':
    train_folder = "./TBD_processed/train"
    dev_folder = "./TBD_processed/dev"
    test_folder = "./TBD_processed/test"

    train_files = os.listdir(train_folder)
    dev_files = os.listdir(dev_folder)
    test_files = os.listdir(test_folder)

    train_result = list()
    dev_result = list()
    test_result = list()

    for file_name in tqdm(train_files):
        file_path = os.path.join(train_folder, file_name)
        train_result.extend(generate(file_path))
    
    for file_name in tqdm(dev_files):
        file_path = os.path.join(dev_folder, file_name)
        dev_result.extend(generate(file_path))
    
    for file_name in tqdm(test_files):
        file_path = os.path.join(test_folder, file_name)
        test_result.extend(generate(file_path))

    with open("./ptuning_data/train.json", mode='w', encoding='utf-8') as fp:
        for r in train_result:
            json_str = json.dumps(r)
            fp.write(json_str)
            fp.write('\n')
    
    with open("./ptuning_data/dev.json", mode='w', encoding='utf-8') as fp:
        for r in dev_result:
            json_str = json.dumps(r)
            fp.write(json_str)
            fp.write('\n')
    
    with open("./ptuning_data/test.json", mode='w', encoding='utf-8') as fp:
        for r in test_result:
            json_str = json.dumps(r)
            fp.write(json_str)
            fp.write('\n')