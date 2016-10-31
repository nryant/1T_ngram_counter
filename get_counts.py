#!/usr/bin/env python
"""Script for collecting ngram counts from the Google Web 1T 5-gram corpus
(LDC2006T13)
"""
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from collections import defaultdict
import gzip
import logging
import os
import sys

from joblib import delayed, Parallel
import numpy as np

from logger import configure_logger

logger = logging.getLogger()
configure_logger(logger)



def load_ngrams(ngramsf, enc='utf-8'):
    """Load n-grams from file.

    Ngrams are stored as strings.
    """
    ngrams = set()
    with open(ngramsf, 'rb') as f:
        for line in f:
            words = line.decode(enc).strip().split()
            n_words = len(words)
            ngram = ' '.join(words)
            if n_words > 5:
                logger.warn('Out-of-range ngram: %s' % ngram.encode('utf-8'))
                continue
            ngrams.add(ngram)
    return ngrams


def load_indexes(corpus_dir):
    """Load ngram indexes.

    Parameters
    ----------
    corpus_dir : str
        Path to LDC206T13

    Returns
    -------
    n_to_indexes : dict
        Mapping from ngram order (an int) to pairs (``fns``, ``init_ngrams``),
        where ``init_ngrams[i]`` is the initial ngram present in ``fns[i]``.
    """
    n_to_indexes = {}
    for n in [1, 2, 3, 4, 5]:
        ngram_dir = os.path.join(corpus_dir, 'data', '%dgms' % n)
        if n == 1:
            fns = [os.path.join(ngram_dir, 'vocab.gz')]
            init_ngrams = ['!']
        else:
            idxf = os.path.join(ngram_dir, '%dgm.idx' % n)
            with open(idxf, 'rb') as f:
                fns = []
                init_ngrams = []
                for line in f:
                    fn, init_ngram = line.decode('utf-8').strip().split('\t')
                    fn = os.path.join(ngram_dir, fn)
                    fns.append(fn)
                    init_ngrams.append(init_ngram)
        fns = np.array(fns)

        init_ngrams = np.array(init_ngrams)
        n_to_indexes[n] = [fns, init_ngrams]
    return n_to_indexes


def get_parent_file(ngram, n_to_indexes):
    """Get file that contains ngram, if present."""
    n = len(ngram.split())
    fns, init_ngrams = n_to_indexes[n]
    ind = np.searchsorted(init_ngrams, ngram)
    try:
        gzf = fns[ind]
    except IndexError:
        gzf = fns[-1]
    return gzf


def partition_ngrams(ngrams, corpus_dir):
    """Partition ngrams by parent file.

    Parameters
    ----------
    ngrams : iterable
        Iterable of ngrams, each a string of whitespace-delimited tokens.

    corpus_dir : str
        Path to LDC206T13.

    Returns
    -------
    gzf_to_ngrams : dict
        Mapping from paths to ``.gz`` files to sets of ngrams to search for
        in those files.
    """
    n_to_indexes = load_indexes(corpus_dir)
    gzf_to_ngrams = defaultdict(set)
    for ngram in ngrams:
        gzf = get_parent_file(ngram, n_to_indexes)
        gzf_to_ngrams[gzf].add(ngram)
    return gzf_to_ngrams


def get_ngram_count(gzf, ngrams):
    """Get ngram counts from a gzipped file.

    Parameters
    ----------
    gzf : str
        Path to gzipped counts file to search.

    ngrams : iterable of str
        Iterable over ngrams in which each ngram occurs once.

    Returns
    -------
    ngram_to_count : dict
        Mapping from ngrams to their counts.
    """
    ngram_to_count = {}
    ngrams_seen = set()
    with gzip.open(gzf, 'r') as f:
        for line in f:
            ngram, n = line.decode('utf-8').strip().split('\t')
            if ngram in ngrams:
                ngram_to_count[ngram] = int(n)
                ngrams_seen.add(ngram)
            if ngrams == ngrams_seen:
                break
    for ngram in (ngrams - set(ngram_to_count.keys())):
        ngram_to_count[ngram] = 0
    return ngram_to_count


def get_ngram_counts(ngramsf, corpus_dir, n_jobs=1):
    """Get counts for all ngrams in a specified file.

    Parameters
    ----------
    ngramsf : str
        Path to file containing ngrams. There should be one ngram per line,
        each a whitespace delimited sequence of words.

    corpus_dir : str
        Path to LDC206T13

    Returns
    -------
    nrgram_to_count : dict
        Mapping from ngrams to counts.
    """
    ngrams = load_ngrams(ngramsf)
    gzf_to_ngrams = partition_ngrams(ngrams, corpus_dir)
    f = delayed(get_ngram_count)
    res = Parallel(n_jobs=n_jobs)(f(*item) for item in gzf_to_ngrams.items())
    ngram_to_count = dict()
    for ngram_to_count_ in res:
        ngram_to_count.update(ngram_to_count_)
    return ngram_to_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment files using Punkt algorithm.',
                                     add_help=False,
                                     usage='%(prog)s [options] ngramsf corpus_dir')
    parser.add_argument('ngramsf', nargs='?',
                        help='Ngrams file.')
    parser.add_argument('corpus_dir', nargs='?',
                        help='Path to LDC206T13.')

    parser.add_argument('--n_jobs', nargs='?', default=1, type=int,
                        metavar='n',
                        help='Number of jobs to run in parallel. (Default: 1)')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ngram_to_count = get_ngram_counts(args.ngramsf, args.corpus_dir,
                                      args.n_jobs)
    for ngram, count in sorted(ngram_to_count.items(),
                               key=lambda x: x[0]):
        logger.info('%s\t%d' % (ngram, count))
