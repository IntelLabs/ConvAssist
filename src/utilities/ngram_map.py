import collections


class NgramMap:
    """
    A memory efficient store for ngrams.

    This class is optimized for memory consumption, it might be slower than
    other ngram stores. It is also optimized for a three step process:

    1) Add all ngrams.
    2) Perform a cutoff opertation (optional).
    3) Read list of ngrams.

    It might not perform well for other use cases.
    """

    def __init__(self):
        """Initialize internal data stores."""
        self._strings = dict()
        self.ngrams = collections.defaultdict(int)
        self.next_index = 0

    def add_token(self, token):
        """
        Add a token to the internal string store.

        This will only add the token to the internal strings store. It will
        return an index that you can use to create your ngram.

        The ngrams a are represented as strings of the indices, so we will
        return a string here so that the consumer does not have to do the
        conversion.

        Parameters
        ----------
        token : str
            The token to add to the string store.

        Returns
        -------
        str
            The index of the token as a string.
        """
        if token in self._strings:
            return str(self._strings[token])
        else:
            self._strings[token] = self.next_index
            old_index = self.next_index
            self.next_index += 1
            return str(old_index)

    def add(self, ngram_indices):
        """
        Add an ngram to the store.

        This will add a list of strings as an ngram to the ngram store. In our
        standard use case the strings are the indices of the strings, you can
        get those from the `add_token()` method.

        Parameters
        ----------
        list of str
            The indices of the ngram strings as string.
        """
        self.ngrams["\t".join(ngram_indices)] += 1

    def cutoff(self, cutoff):
        """
        Perform a cutoff on the ngram store.

        This will remove all ngrams that have a frequency with the given cutoff
        or lower.

        Parameters
        ----------
        cutoff : int
            The cutoff value, we will remove all items with a frequency of the
            cutoff or lower.
        """
        delete_keys = []
        for k, count in self.ngrams.items():
            if count <= cutoff:
                delete_keys.append(k)
        for k in delete_keys:
            del self.ngrams[k]

    def __len__(self):
        """Return the number of ngrams in the store."""
        return len(self.ngrams)

    def items(self):
        """
        Get the ngrams from the store.

        Returns
        -------
        iterable of tokens, count
            The tokens are a list of strings, the real tokens that you added
            to the store via `add_token()`. The count is the the count value
            for that ngram.
        """
        strings = {v: k for k, v in self._strings.items()}
        for token_indices, count in self.ngrams.items():
            tokens = [strings[int(idx)] for idx in token_indices.split("\t")]
            yield tokens, count