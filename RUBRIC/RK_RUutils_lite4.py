#!/usr/bin/env python
import csv
import os
import sys

from Bio import SeqIO


def file_len(file_name):
    """
    calculates length of file
    """
    with open(file_name) as f:
        # print 'file is' + str(fname)
        for i, l in enumerate(f):
            pass
    return i + 1


def align_reads(aligner, db, in_seq, in_seq_id='noIDgiven', offline=False, number_of_bases=0, verbose=False):
    """
    align the reads using graphmap or last
    if working offline, remember to add 'offline=True' otherwise the script will error out
    also if using last and there is already a database, toggle isdb to true
    """
    current_path = os.getcwd()
    if offline:
        # print 'offline'
        if number_of_bases == 0:
            sys.exit('you are in offline mode but have not specified number of bases - please fix')
        else:
            counter = 0
            for record in SeqIO.parse(in_seq, "fasta"):
                counter += 1
                # print 'count is ' + str(counter) ##THIS IS FOR DIAGNOSTICS
                new_seq = record.seq
                new_seq2 = str(new_seq[0:int(number_of_bases) + 1])
                if aligner == 'graphmap':
                    graph_map(new_seq2, record.id, current_path, db, verbose)
                elif aligner == 'last':
                    # print 'yes database'
                    result = last(new_seq2, record.id, current_path, db, verbose)
                    return result
                else:
                    sys.exit('invalid aligner specified in test mode - use either graphmap or last')
    else:
        #        print 'realtime'
        if in_seq_id == 'noIDgiven':
            sys.exit('no read ID given - if not running in offline mode, please pass read ID as third arguemnt')
        else:
            if aligner == 'graphmap':
                graph_map(in_seq, in_seq_id, current_path, db, verbose=False)
            elif aligner == 'last':
                result = last(in_seq, in_seq_id, current_path, db, verbose=False)
                # print 'made it here '
                # print result
                return result
            #                else:
            #                    if dbName=='noNamegiven':
            #                        #print 'no database name given'
            #                        sys.exit('please specify database name for last')
            #                    else:
            #                        #print 'databse name given'
            #                        Lastdb(db,dbName)
            #                        #print 'databse is ' + str(dbName)
            #                        result=Last(in_seq,in_seq_id,current_path ,dbName,verbose=False)
            #                        #print 'made it here '
            #                        #print result
            #                        return result
            else:
                sys.exit('invalid aligner specified in realtime mode - use either graphmap or last')


def graph_map(sequence, id, current_path, db, verbose):
    """
    given a sequence and ID, map using graphmap
    """
    new_fasta = ">" + str(id) + "\n" + str(sequence)
    tempfile = open(str(id) + ".fa", "w")
    tempfile.write(new_fasta)
    tempfile.close()
    in_fasta = str(current_path) + "/" + str(id) + ".fa"
    name = 'tmpOutGrM' + str(id)
    cmdstring = "graphmap align -r %s -d %s -o %s -v 0 -a anchor -z 0.5" % (db, in_fasta, name)
    os.system(cmdstring)
    graph_out = str(current_path) + "/" + str(name)
    f = open(graph_out, 'r')
    sam_file = f.read()
    samlist = sam_file.split('\n')
    for a in samlist:
        if a[0] != "@":
            flagline = a
            break
    flag = flagline.split('\t')[1]
    # print 'flag is '+str(flag) ##THIS IS FOR DIAGNOSTICS
    sam_check(name, flag, verbose)
    os.remove(name)
    os.remove(in_fasta)


def last_db(input_fasta, db_name):
    """
    input fasta, output database for last alignment
    in db_name, include path otherwise will be put into current folder.
    for inFasta include path if not in current folder
    """
    cmdstring = "lastdb -cR01 %s %s" % (db_name, input_fasta)
    # print cmdstring
    os.system(cmdstring)


def last(sequence, id, current_path, db, verbose):
    """
    given a fasta and database, output an alignment file
    """
    newFasta = ">" + str(id) + "\n" + str(sequence)
    tempfile = open(str(id) + ".fa", "w")
    tempfile.write(newFasta)
    tempfile.close()
    inFasta = str(current_path) + "\\" + str(id) + ".fa"
    #    dbName=str(currpath)+"/"+str(db)
    name = 'tmpOutGrM' + str(id)
    cmdstring = "lastal -fTAB -C2 %s %s > %s" % (db, inFasta, name)
    os.system(cmdstring)
    last_out = str(current_path) + "\\" + str(name)
    #    print 'last_out is:'

    if file_len(last_out) == 20:
        os.remove(last_out)
        os.remove(inFasta)
        return 'Skip'
    else:
        os.remove(last_out)
        os.remove(inFasta)
        return 'Sequence'


def last_batch(file, currpath, db, cmdstring):
    """
    takes a batch input in fasta format and outputs a dictionary of calls. also provide current path and databse
    """
    inFasta = str(currpath) + "\\" + str(file)
    updatedDict = {}
    with open(inFasta) as a:
        reader = csv.reader(a, delimiter="\n")
        c = list(reader)
        # print 'we are in lastbatch, len of list is ',len(c)
        for a in range(0, len(c) - 1):
            if a % 2 == 0:
                # print 'here is a ',c[a]
                # default all channels to skip
                updatedDict[c[a][0].split("_")[0].replace('>', '')] = [c[a][0].split("_")[1], "Skip"]
                # name=c[a][0]
                # channel=name.split("_")[0]
                # read=name.split("_")[1]
    name = 'tmpOutGrM' + str(file)
    #    cmdstring="lastal -fTAB -C2 -q 1 -r 1 -a 1 -b 1 -D 100 -e 15 %s %s > %s" % (db,inFasta,name)
    cmdstring = 'lastal' + cmdstring + '%s %s > %s' % (db, inFasta, name)
    os.system(cmdstring)
    last_out = str(currpath) + "\\" + str(name)
    with open(last_out) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)
        for i in range(19, len(d) - 1):
            name = d[i][6]
            read = name.split("_")[1]
            channel = name.split("_")[0].replace('>', '')
            # this if statement is a sanity check - can be removed to save time
            if channel in updatedDict.keys() and updatedDict[channel][0] == read:
                updatedDict[channel][1] = "Sequence"
            else:
                sys.exit('something is very wrong')
    return updatedDict


def sam_check(name, flag, verbose):
    if flag == '4':
        print 'Skip'
        if verbose:
            print name
    elif flag == '0' or flag == '16':
        print 'Sequence'
        if verbose:
            print name
    else:
        if verbose:
            print name
        print "What is this??? don't be lazy Raga go check it NOW!!!"

