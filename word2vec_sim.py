##
## author - Simone Marchi (simone.marchi@ilc.cnr.it)
##
## Script to use gensim word2vec library for training models
## and search similars to a specific word
##
import pandas as pd
from gensim.models import Word2Vec
import os
import os.path
import getopt, sys
import tempfile
from sys import stdin
import datetime
import logging

logger = logging.getLogger('w2v')

def usage():
    print ("""usage: python {} -w|--word word_to_search
    \t-t|--train directory_of_training_set
    \t-r|--run model_file_name
    \t-w|--windowsize window_size (default = 10)
    \t-v|--vectorsize  vector_size(default = 50)
    \t-e|--epoch numer_of_epoch (default = 20)
    \t-x|--extension (default: any files) [e.g. txt]
    \t-a|--algorithm 1 for skip-gram, 0 for CBOW (default 0:CBOW)
    \t-l|--loglevel integer (0: not set, 10: debug, 20: info, 30: warning, 40: error, 50: critical)
    \t-h|--help
    \t Example: python {} """.format(sys.argv[0],sys.argv[0]))
    sys.exit()

def trainModel (trainDir, extension, _vector_size:int, _window:int, _epochs:int, algorithm:int):

    data = []
    logger.debug("Training directory is {}".format(trainDir))
    for dirpath, dirnames, filenames in os.walk(trainDir):
        #looking for specified filenames
        for filename in [f for f in filenames if f.endswith(extension)]:
            try:
                with open(os.path.join(dirpath, filename)) as file:
                    lines = file.readlines()
                    for sent in lines:
                        data.append(sent.split())
                    logger.debug("Read {}".format(filename))
                    file.close()
            except UnicodeDecodeError as e:
                logger.info("Cannot read file {}".format(filename))
    logger.info ("number of tokens: {}".format(len(data)))

    if len(data) == 0:
        logger.error("no tokens found in directory {}".format(trainDir))
        exit()
    logger.info("Creating Word2Vec")
    startTime = datetime.datetime.now()
    model = Word2Vec(
        sentences = data,
        vector_size = _vector_size ,
        window = _window,
        epochs = _epochs,
        sg = algorithm,
        min_count = 5,
        workers = 8
    )
    endTime = datetime.datetime.now()

    modelfilename = "model-vs{}-w{}-e{}-a{}.bin".format(_vector_size,_window,_epochs,algorithm)
    with open(modelfilename, 'wb') as outfile:
        model.save(outfile)
        logger.info("Model {} trained in {}".format(modelfilename,endTime-startTime))
    logger.info("Training is done")


def main():
    type = None
    word = None
    trainDir = None
    modelFilename = None
    vectorsize = 50
    window = 10
    epochs = 20
    algorithm = 0 #0: CBOW, 1: skip-gram
    logLevel = 20 #log level set to INFO
    extension = ""

    opts, args = getopt.getopt(sys.argv[1:],
                               't:r:w:v:e:a:x:l:h',
                               ["train=", "run=", "windowsize=",
                                "vectorsize=", "epochs=",
                                "directory=", "algorithm=",
                                "extension","loglevel","help"])

     #Set up the logging system
    logFormatter = logging.Formatter(fmt=' %(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s',
                                     datefmt='%d.%m.%y %H:%M:%S')
    consoleHandler = logging.StreamHandler(sys.stderr)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.debug("Arguments {}".format(sys.argv[1:]))

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-r", "--run"):
            if type is None:
                type = "run"
                modelFilename = a
            else:
                logger.error("You cannot specify run and training at the same time")
                usage()
        elif o in ("-t", "--train"):
            if type is None:
                type = "train"
                trainDir = a #input and output directory are thae same
            else:
                logger.error("You cannot specify run and training at the same time")
                usage()
        elif o in ("-w", "--window" ):
            window = int(a)
        elif o in ("-v", "--vectorsize" ):
            vectorsize = int(a)
        elif o in ("-e", "--epochs" ):
            epochs = int(a)
        elif o in ("-a", "--algorithm" ):
            algorithm = int(a) #sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
            if algorithm not in (0,1):
                logger.error("ERR: Training algorithm: 1 for skip-gram, 0 for CBOW.")
                usage()
        elif o in ("-x", "--extension" ):
            extension = a
        elif o in ("-l", "--loglevel"):
            try:
                logLevel = int(a)
                if (logLevel not in [0, 10, 20, 30, 40, 50]):
                    raise ValueError("Log level must be: 0, 10, 20, 30, 40 or 50")
                logger.setLevel(logLevel)
            except ValueError as ve:
                logger.error("Error: {}".format(ve))
                usage()
        else:
            logger.critical("unhandled option")

    if type == None:
        logger.error("You must specify trainig or running mode")
        usage()

    if type == "run":
        logger.info("Loading model...", file=sys.stderr, flush=True)
        if os.path.exists(modelFilename):
            model = Word2Vec.load(modelFilename)
        else:
            logger.error("ERR: model {} doesn't exist".foramt(modelFilename))
            exit();
        logger.info("Model loaded {}".format(len(model.wv)))

        # Finding most similar words
        print("Enter Exit/Quit to close")
        print(">>> ", end='',flush=True)
        for token in sys.stdin:
            token = token.strip()
            if token in ["Exit", "exit","Quit","quit","q"]:
                break
            try:
                most_similar = model.wv.most_similar( token, topn=10 )
                print ("Similar to", token)
                for term, score in most_similar:
                    print(term, score)
            except KeyError as e:
                print("Word {} not found".format(token))
            print(f'Processing Message from sys.stdin *****{token}*****')
            print(">>> ", end='',flush=True)
        print("Done")

    else:
        logger.info("Training configuration: vectorsize: {}, window: {}, epochs: {}, algorithm: {}".format(vectorsize,window,epochs,algorithm))
        trainModel(trainDir, extension, vectorsize , window, epochs, algorithm)

if __name__ == '__main__':
    main()
