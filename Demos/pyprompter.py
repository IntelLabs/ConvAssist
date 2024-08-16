import sys
import getopt
import os
import configparser
PROGRAM_NAME = 'pyprompter'
config = None
suggestions = None

def print_version():
    print("""
%s (%s) version %s
Copyright (C) 2004 Matteo Vescovi.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE,
to the extent permitted by law.
""" % (PROGRAM_NAME, 'presage', '0.9.2~beta'))


def print_usage():
    print("""
Usage: %s [options]

Options:
  -h, --help           display this help and exit
  -v, --version        output version information and exit

Prompter is a simple predictive text editor, written to demonstrate
the use of presage, the intelligent predictive text entry system.

Begin editing. While editing, prompter uses presage to generate
predictions and displays them in a pop-up prediction list. If the
desired text is displayed in the prediction list, select it by double
clicking on it or by highlighting it with the arrow keys and then
pressing ENTER; the desired text will be automatically entered.

Direct your bug reports to: %s
""" % (PROGRAM_NAME, 'matteo.vescovi@yahoo.co.uk'))


def parse_cmd_line_args():
    short_options = "c:s:hv"
    long_options  = ["config=", "suggestions=", "help", "version"]

    try:
        opts, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-v', '--version'):
            print_version()
            sys.exit()
        elif opt in ('-h', '--help'):
            print_usage()
            sys.exit()
        elif opt in ('-c', '--config'):
            global config
            config = arg
        elif opt in ('-s', '--suggestions'):
            global suggestions
            suggestions = arg


if __name__ == "__main__":
    parse_cmd_line_args()

    try:
        #import prompter.prompter
        import prompter
    except ImportError as e:
        print('''
Error: failed to import module prompter.

Check that prompter is properly installed (if installed in a
non-standard location, please set PYTHONPATH accordingly).
''')
        print(e)
    else:
        if not config:
            SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

            config_file = os.path.join(SCRIPT_DIR, "wordPredMode.ini")
            # config_file = os.path.join(SCRIPT_DIR, "shortHandMode.ini")
            # config_file = os.path.join(SCRIPT_DIR, "cannedPhrasesMode.ini")
            config = configparser.ConfigParser()
            config.read(config_file)

        #app = prompter.prompter.Prompter("0.9.2~beta", config, suggestions)
        app = prompter.Prompter("0.9.2~beta", config, suggestions)
        app.MainLoop()
