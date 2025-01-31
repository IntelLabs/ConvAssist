import argparse
from convassist.utilities.ngram.ngramutil import NGramUtil

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
       '--cardinality',
        type=int,
        default=3,
        help='The number of tokens to consider in the n-gram model'
    )

    parser.add_argument(
        "--lowercase",
        type=bool,
        default=False,
        help="Whether to convert all tokens to lowercase"
    )

    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Whether to normalize the database"
    )
    return parser

def main(argv=None):
    parser = configure()
    args = parser.parse_args(argv)
    
    with NGramUtil(args.database, args.cardinality, args.lowercase, args.normalize) as ngramutil:
        phrases = []

        with open(args.input_file) as f:
            for line in f:
                phrases.append(line.strip())

        ngramutil.update(phrases)

if __name__ == "__main__":
    main()