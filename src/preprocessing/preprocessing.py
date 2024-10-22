from config import * 

import re
import json

valid_pattern = r'^[a-zA-Z0-9\s.,?!;:\'"()\\-]*$'
tokenize_reg = r'(\s|,|!|\?|\.|:|;|\'|\"|“|”|‘|’|\(|\)|\[|\]|\{|\})'

def is_text_clean(s: str) -> bool:
    if s is None: 
        return False

    if '###>' not in s: 
        return False

    s = s.lower()

    ind_sentence, eng_sentence = s.split('###>', 1)

    if len(ind_sentence) <= 0 or len(eng_sentence) <= 0:
        return False
    
    ind_sentence = ind_sentence.strip()
    eng_sentence = eng_sentence.strip()
    
    if not re.match(valid_pattern, ind_sentence) or not re.match(valid_pattern, eng_sentence):
        print("NOT CLEAN:", ind_sentence, eng_sentence)
        return False 

    return True

def tokenize(s: str) -> map:
    """Tokenize a string

    Args:
        s (str): String containing a sentence written in both
                 Indonesian and English, separated by '###>'.
                 The string should be clean. Uncleaned data samples
                 should be filtered out using text_is_clean().

    Returns:
        map: Map with keys "id" and "eng", with their values being
             a list of their respective tokens.
    """

    ret = {"id": list(), "eng": list()}

    id, eng = [*s.split("###>")]

    # id
    tks = re.split(tokenize_reg, id.strip()) # split with various symbols (excluding hyphens)
    for tk in tks:
        if tk.replace(' ', '') != '':
            ret["id"].append(tk)
    # eng
    tks = re.split(tokenize_reg, eng.strip())
    for tk in tks:
        if tk.replace(' ', '') != '':
            ret["eng"].append(tk)

    ret["id"].append(EOS)
    ret["eng"].append(EOS)

    # print(ret)
    return ret

def tokenize_english(s: str) -> list:
    ret = []
    tks = re.split(tokenize_reg, s)
    for tk in tks:
        if tk.replace(' ', '') != '':
            ret.append(tk)
    return ret

def save_vocab_to_disk(tokens: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(tokens, f)

def load_vocab_from_disk(filename: str) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)
