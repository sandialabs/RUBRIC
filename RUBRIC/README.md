***************************************************************************
(c) 2018: National Technology & Engineering Solutions of Sandia, LLC (NTESS)
***************************************************************************

RUBRIC
======
Read Until with Basecall and Reference-Informed Criteria

These scripts allow for real-time filtering of nanopore sequencing reads based upon analysis of the incoming basepairs as in [DOI](https://www.biorxiv.org/content/early/2018/11/02/460014).  RUBRIC was conceived and tested using Nanonet for basecalling, LAST for read-alignment, and a Windows 10 PC operating in safe mode (with networking).

NOTE:   
---
A version of nanonet has been included in this distribution which is no longer offered or supported by Oxford Nanopore Technologies (ONT).  It has been included to maintain the functionality of this package. 

Also note that it is advisable to use this package first on a used flow cell to gauge baseline functionality. 

Installation
------------
**Requirements**

As the Nanonet basecaller is no longer supported or offered by Oxford Nanopore Technologies (ONT), a version modified to support RUBRIC has been included in this repository. 
RUBRIC also relies upon the [LAST](http://last.cbrc.jp/) aligner.  Therefore LAST must also be installed and added to PATH. Nanonet will need to be compiled using the [Visual C++ Compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266).  RUBRIC relies upon the Read-Until API, which can be obtained directly from ONT.  Most of the results obtained in [DOI](https://www.biorxiv.org/content/early/2018/11/02/460014) were obtained using the RU API that was released alongside MinKNOW version 1.6.11.  *RUBRIC has not been tested on newer versions of the RU API, but may work with some small adjustments.* 

**Install**

Once LAST and the C++ compiler have been installed the RUBRIC scripts can be installed.  It is highly recommended to install the scripts in a virtual environment such as conda:
``` 
conda create -n RUBRIC_env python=2.7

activate RUBRIC_env

cd \path\to\cloned\repository

python setup.py install   
```

This setup file first installs the RUBRIC components, and then calls the 'setup_nanonet.py' file (taken and renamed from the original Nanonet repository).   

RUBRIC relies on an older version of the read_until API, which is included in this repository and is used via a relative import during runtime.  It is recommended that users with the newer read_until API first uninstall the new version before installing RUBRIC. 


Quick Start
-----------
Once installed, the rubric commandline help can be called via RUBRIC can be called simply with:

```
python RUBRIC -h
```

Which should then show:
```
usage: RUBRIC [-h] -r REFERENCE_DATABASE -ho HOST [-a ALIGN]
              [-as ALIGNER_SETTINGS] [-t TIME] [-q QUEUE_SIZE] [-s]
              [-l LOWER_THRESHOLD] [-u UPPER_THRESHOLD] [-i IGNORE_EVENTS]
              [-c CONSIDER_EVENTS]

Read Until with Basecall and Reference-Informed Criteria (RUBRIC)

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE_DATABASE, --reference_database REFERENCE_DATABASE
                        path to database if LAST or fasta file if graphmap
  -ho HOST, --host HOST
                        The host address for the laptop running the MinION
  -a ALIGN, --aligner ALIGN
                        Type of aligner - either "graphmap" or "last" (default
                        last)
  -as ALIGNER_SETTINGS, --aligner_settings ALIGNER_SETTINGS
                        A string containing the settings to pass to the
                        aligner (default: '-fTAB -C2 -q 1 -r 1 -a 1 -b 1 -e
                        30'
  -t TIME, --time TIME  This is an error catch for when we cannot keep up with
                        the rate of sequencing on the device. It takes a
                        finite amount of time to process through the all the
                        channels from the sequencer. If we cannot process
                        through the array quickly enough then we will 'fall
                        behind' and lose the ability to filter sequences.
                        Rather than do that we set a threshold after which we
                        allow the sequencing to complete naturally.
  -q QUEUE_SIZE, --queue QUEUE_SIZE
                        The length of the queue for storing reads until
                        compute resources are available. (default 16)
  -s, --skip_even       If set, only apply filtering to even pores
  -l LOWER_THRESHOLD, --lower_threshold LOWER_THRESHOLD
                        The lower standard deviation threshold to filter reads
                        before basecalling (default 5)
  -u UPPER_THRESHOLD, --upper_threshold UPPER_THRESHOLD
                        The upper standard deviation threshold to filter reads
                        before basecalling (default 14)
  -i IGNORE_EVENTS, --ignore_events IGNORE_EVENTS
                        The number of events to ignore at the beginning of the
                        read (default 100)
  -c CONSIDER_EVENTS, --consider_events CONSIDER_EVENTS
                        The number of events to after the ignored events to be
                        used for RUBRIC consideration (default 300)
```

**Required Arguments**

Only the path to the reference database and the host address are required arguments.  All other arguments default to empirically-determined optimal conditions observed in [DOI].  After ensuring that the event sampler has started with MinKNOW, one can simply use:

```
python RUBRIC --reference_database \path\to\LAST\database --host ws://localhost:9200/  
```

Note the above command assumes the event sampler is running locally on port 9200.  It is highly desireable to have one computer running MinKNOW (and the event sampler) and one computer that connects remotely and runs RUBRIC.  



