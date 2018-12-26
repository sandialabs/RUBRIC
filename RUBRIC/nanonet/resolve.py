import csv
from datetime import datetime
import glob
import os
import re
import sys

reads_dir = sys.argv[1]

sampler_reads = {str(n+1): {} for n in range(512)}
with open('read_log.csv', 'rb') as log_file:
    reader = csv.reader(log_file)
    reader.next() # skip header
    i = 0
    for row in reader:
        [date_str, channel_name, read_number, delay] = row
        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        sampler_reads[channel_name][read_number] = date

with open('missing_files.csv', 'wb') as missing_f, open('matched_files.csv', 'wb') as matched_f:
    missing = csv.writer(missing_f)
    missing.writerow(["Channel", "Read", "File"])
    matched = csv.writer(matched_f)
    matched.writerow(["Channel", "Read", "Time from seen to written", "File"])

    regex = re.compile('.*_ch([0-9]+)_read([0-9]+)_.*\\.fast5')
    for (dirpath, dirnames, filenames) in os.walk(reads_dir):
        for filename in filenames:
            match = regex.match(filename)
            if match:
                channel_name = match.group(1)
                read_number = match.group(2)
                pretty_path = os.path.join(os.path.basename(dirpath), filename)
                try:
                    date = sampler_reads[channel_name][read_number]
                except KeyError:
                    missing.writerow([channel_name, read_number, pretty_path])
                else:
                    path = os.path.join(dirpath, filename)
                    last_mod = datetime.fromtimestamp(os.path.getmtime(path))
                    matched.writerow([channel_name, read_number, str(last_mod - date), pretty_path])
