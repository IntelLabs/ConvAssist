import re
import typing
from ConvAssist.utilities.ngram_map import NgramMap
from ConvAssist.tokenizer.forward_tokenizer import ForwardTokenizer

def preprocess(text):
    re_wordbeg = re.compile(r"(?<=\s)[-']")
    re_wordbeg2 = re.compile(r"(?<=\s\")[-']")
    re_wordend = re.compile(r"[-'](?=\s)")
    re_wordend2 = re.compile(r"[-'](?=\"\s)")
    text = re_wordbeg.sub("", text)
    text = re_wordbeg2.sub("", text)
    text = re_wordend.sub("", text)
    text = re_wordend2.sub("", text)
    return text

def forward_tokenize_files(
    infiles: typing.List[str], ngram_size: int, lowercase: bool = False, cutoff: int = 0
):
    """
    Tokenize a list of file and return an ngram store.

    Parameters
    ----------
    infile : str
        The file to parse.
    ngram_size : int
        The size of the ngrams to generate.
    lowercase : bool
        Whether or not to lowercase all tokens.
    cutoff : int
        Perform a cutoff after parsing. We will only return ngrams that have a
        frequency higher than the cutoff.

    Returns
    -------
    NgramMap
        The ngram map that allows you to iterate over the ngrams.
    """
    ngram_map = NgramMap()
    for infile in infiles:
        ngram_map = forward_tokenize_file(
            infile, ngram_size, lowercase=lowercase, ngram_map=ngram_map
        )

    if cutoff > 0:
        ngram_map.cutoff(cutoff)

    return ngram_map

def forward_tokenize_file(
    infile: str,
    ngram_size: int,
    lowercase: bool = False,
    cutoff: int = 0,
    ngram_map: NgramMap = None,
):
    """
    Tokenize a file and return an ngram store.

    Parameters
    ----------
    infile : str
        The file to parse.
    ngram_size : int
        The size of the ngrams to generate.
    lowercase : bool
        Whether or not to lowercase all tokens.
    cutoff : int
        Perform a cutoff after parsing. We will only return ngrams that have a
        frequency higher than the cutoff.
    ngram_map : NgramMap
        Pass an existing NgramMap if you want to add the ngrams of the given
        file to the store. Will create a new NgramMap if `None`.

    Returns
    -------
    NgramMap
        The ngram map that allows you to iterate over the ngrams.
    """
    if ngram_map is None:
        ngram_map = NgramMap()

    with open(infile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = preprocess(line)
            ngram_list = []
            tokenizer = ForwardTokenizer(line)
            tokenizer.lowercase = lowercase
            while len(ngram_list) < ngram_size - 1 and tokenizer.has_more_tokens():
                token = tokenizer.next_token()
                if token != "":
                    token_idx = ngram_map.add_token(token)
                    ngram_list.append(token_idx)
            if len(ngram_list) < ngram_size - 1:
                continue

            tokenizer.reset_stream()
            while tokenizer.has_more_tokens():
                token = tokenizer.next_token()
                if token != "":
                    token_idx = ngram_map.add_token(token)
                    ngram_list.append(token_idx)
                    ngram_map.add(ngram_list)
                    ngram_list.pop(0)

    if cutoff > 0:
        ngram_map.cutoff(cutoff)

    return ngram_map