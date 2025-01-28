import argparse
from typing import List

from tqdm import tqdm
from convassist.utilities.ngram.ngramutil import NGramUtil
from convassist.utilities.ngram.ngram_map import NgramMap

# from multiprocessing import Process, Thread
from threading import Thread


def configure():
    # Create top-level parser
    parser = argparse.ArgumentParser(description="Recreate an NGram Database for ConvAssist")

    # File command
    parser.add_argument(
        "database",
        type=str,
        help="The database file to use for the n-gram model. Must be a path to a db file."
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='The input file to use for the database.  Must be a text file with one sentence per line.'
    )

    parser.add_argument(
        '-c',
        '--cardinality',
        type=int,
        default=3,
        help='The number of tokens to consider in the n-gram model'
    )

    parser.add_argument(
        '-l',
        "--lowercase",
        action="store_true",
        help="Whether to convert all tokens to lowercase"
    )

    parser.add_argument(
        '-n',
        "--normalize",
        action="store_true",
        help="Whether to normalize the database"
    )

    #flag to clean the database

    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Whether to clean the database"
    )

    return parser


def insertngrambycardinality(ngramutil: NgramMap, phrases: List, cardinality: int):

    query = ngramutil.generate_ngram_insert_query(cardinality, True)

    data = []
    for phrase in tqdm(phrases, desc=f"Processing {cardinality}-grams", unit=" phrases", leave=False):
        ngram_map = NgramMap(cardinality, phrase)
        for ngram, count in ngram_map.items():
            data.append((*ngram, count))

    batch_size = 3000
    for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting {cardinality}-grams", unit=" batches", leave=True):
        batch = data[i:i + batch_size]
        ngramutil.connection.execute_many(query, batch)


def main(argv=None):
    parser = configure()
    args = parser.parse_args(argv)

    if args.clean:
        # give a warning that the database will be deleted and recreated
        print(f"Cleaning database {args.database}")
        print("This will delete the database and recreate it from scratch.")
        print("Are you sure you want to continue? (y/n)")
        response = input()
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    phrases = []

    with open(args.input_file) as f:
        for line in f:
            phrases.append(line.strip())

    with NGramUtil(args.database, args.cardinality, args.lowercase, args.normalize) as ngramutil:
        threads = []
        for i in range(args.cardinality):
            p = Thread(target=insertngrambycardinality, args=(ngramutil, phrases, i + 1))
            threads.append(p)
            p.start()

        for p in threads:
            p.join()


if __name__ == "__main__":
    main()
