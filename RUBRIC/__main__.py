import logging
import multiprocessing
import os
import threading
import time
from functools import partial
from multiprocessing import Queue
from socket import error as socket_error
from threading import Thread

import configargparse
import h5py
import numpy as np
from termcolor import colored

import nanonet as nnc
from RK_RUutils_lite4 import last_batch
from read_until.read_until import ReadUntil


def get_args():
    """
    Gets the arguments from the command line.
    """
    p = configargparse.ArgParser(description='Read Until with Basecall and Reference-Informed Criteria (RUBRIC)')
    p.add('-r', '--reference_database', required=True, help='path to database if LAST or fasta file if graphmap',
          dest='reference_database')
    p.add('-ho', '--host', required=True, help='The host address for the laptop running the MinION', dest='host')
    p.add('-a', '--aligner', required=False, help='Type of aligner - either "graphmap" or "last" (default last)',
          dest='align', default='last')
    p.add('-as', '--aligner_settings', required=False,
          help='A string containing the settings to pass to the aligner (default: \'-fTAB -C2 -q 1 -r 1 -a 1 -b 1 -e '
               '30\'',
          dest='aligner_settings', default='-fTAB -C2 -q 1 -r 1 -a 1 -b 1 -e 30')
    p.add('-t', '--time', type=int, dest='time', required=False, default=2,
          help='This is an error catch for when we cannot keep up with the rate of sequencing on the device. It takes '
               'a finite amount of time to process through the all the channels from the sequencer. If we cannot '
               'process through the array quickly enough then we will \'fall behind\' and lose the ability to filter '
               'sequences. Rather than do that we set a threshold after which we allow the sequencing to complete '
               'naturally.')
    p.add('-q', '--queue', required=False, help='The length of the queue for storing reads until compute resources are '
                                                'available. (default 16)', dest='queue_size', default=16)
    p.add('-s', '--skip_even', required=False, action='store_true', help='If set, only apply filtering to even pores',
          dest='skip')
    p.add('-l', '--lower_threshold', required=False, help='The lower standard deviation threshold to filter reads '
                                                          'before basecalling (default 5)', dest='lower_threshold',
          default=5)
    p.add('-u', '--upper_threshold', required=False, help='The upper standard deviation threshold to filter reads '
                                                          'before basecalling (default 14)', dest='upper_threshold',
          default=14)
    p.add('-i', '--ignore_events', required=False, help='The number of events to ignore at the beginning of the read '
                                                        '(default 100)', dest='ignore_events', default=100)
    p.add('-c', '--consider_events', required=False, help='The number of events to after the ignored events to be '
                                                          'used for RUBRIC consideration (default 300)',
          dest='consider_events', default=300)
    run_args = p.parse_args()
    return run_args


def convert_to_event_array(data_event):
    """
    This function takes the data from the event sampler and puts it into the necessary shape for Nanonet.
    :param data_event:
    :return: A structured array in the proper format for Nanonet basecalling
    """
    events_for_sequencing = np.zeros((len(data_event.events),),
                                     dtype=[('start', '<f8'), ('length', '<f8'), ('mean', '<f8'), ('stdv', '<f8')])
    for i in range(len(data_event.events)):
        events_for_sequencing['start'][i] = data_event.events[i].start
        events_for_sequencing['length'][i] = data_event.events[i].length
        events_for_sequencing['mean'][i] = data_event.events[i].mean
        events_for_sequencing['stdv'][i] = data_event.events[i].sd
    return events_for_sequencing


class LockedDict(dict):
    """
    A dict where __setitem__ is synchronised with a new function to
    atomically pop and clear the map.
    """

    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        super(LockedDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        with self.lock:
            super(LockedDict, self).__setitem__(key, value)

    def pop_all_and_clear(self):
        with self.lock:
            d = dict(self)  # take copy as a normal dict
            super(LockedDict, self).clear()
            return d


def create_events_dataset((events, channel_name, read_number)):
    """
    This function will write event data into fast5 files in the same format as other read fast5 files (for easy
    comparison later.
    """
    file_name = 'saved_events\\event_sampled__data_%s_%s_strand.fast5' % (channel_name, read_number)
    f = h5py.File(file_name, 'w')
    f.create_dataset('Analyses/EventDetection_000/Reads/Read_' + str(read_number) + '/Events', data=events)
    f.close()


def get_time():
    return time.time()


def enforce_thresholds(channels):
    """
    This function filters the channel data from the event sampler based upon standard deviation thresholds.
    :param channels: A dictionary of read data from the event sampler
    :return: A dictonary of the filtered read data
    """
    filtered_channels = {}
    for channel_name, data in channels.iteritems():
        temp_sd = np.std([events.mean for events in data.events])
        decision_time = str(time.asctime()).replace(':', '.').replace(' ', '_')
        if args.lower_threshold < temp_sd < args.upper_threshold:
            logging.info('ch%s_rd%s, at std of %s found to be inside of thresholds, time is %s' % (
                channel_name, data.read_number, temp_sd, decision_time))
            filtered_channels[channel_name] = channels[channel_name]
        else:
            logging.info('ch%s_rd%s, at std of %s found to be outside of thresholds, time is %s' % (
                channel_name, data.read_number, temp_sd, decision_time))
            # self.current_unblock_map[channel_name] = int(data.read_number) print 'OUT OF THRESHOLD: ',
            # colored('\tch{0}_rd{1}'.format(channel_name,data.read_number,temp_sd),'red'),' time is : ',
            # decision_time
    return filtered_channels


class RUBRIC:
    """
    Baseline analyzer class for RUBRIC
    """

    def __init__(self, args):
        self.current_unblock_map = LockedDict()
        self.p = multiprocessing.Pool(multiprocessing.cpu_count())  # max it out!
        self.args = args
        self.queue_size = args.queue_size
        self.q = Queue()
        self.threshold_time = args.time
        self.old_reject_decision_dict = {}
        self.old_sequence_decision_dict = {}
        self.t = Thread(target=self.manage_queue)
        self.t.daemon = True
        self.t.start()

    def make_decision(self, decisions, time_then):
        """
        make decision on processed events using LastBatch output dict
        :param decisions:
        :param time_then:
        """
        decision_string = 'Decisions:'
        decision_time = str(get_time() - time_then)
        queue_utilization = str(len(decisions)) + '/' + str(self.queue_size)
        for decision in decisions:
            if decisions[decision][1] == "Skip":
                self.current_unblock_map[decision] = int(decisions[decision][0])
                self.old_reject_decision_dict[decision] = int(decisions[decision][0])
                decision_string += colored('\tch{0}_rd{1}'.format(decision, decisions[decision][0]), 'red')
            else:
                # create_events_dataset((convert_to_event_array(channels[decision]),decision,channels[
                # decision].read_number))
                self.old_sequence_decision_dict[decision] = int(decisions[decision][0])
                decision_string += colored('\tch{0}_rd{1}'.format(decision, decisions[decision][0]), 'green')
            logging.info('\tchannel %s\tread# %s\tdecision %s\t\ttime to decision %s\t time: %s ', decision,
                         decisions[decision][0],
                         decisions[decision][1], decision_time,
                         str(time.asctime()).replace(':', '.').replace(' ', '_'))
        print decision_string, '\t', colored(decision_time, 'yellow'), '\t', colored(queue_utilization, 'cyan')

    def process_events(self, channels_and_time, reference_database):
        """
        manage the basecall on multiple processes here, then pass it to the
        aligner
        :param channels_and_time: A tuple containing the dictionary from the event sampler, and then time when the
        dictionary was given
        :param reference_database: A string containing the location of the LAST database
        """
        channels = channels_and_time[1]
        read_time = channels_and_time[0]
        id_data_list = []

        for channelName, data in channels.iteritems():
            id_data_list.append((channelName, data.read_number, convert_to_event_array(data)))
        if len(id_data_list) > self.queue_size:
            id_data_list = id_data_list[0:self.queue_size]

        # these lines are only necessary if randomization of read analysis is desired
        # templist = []
        # for i in xrange(self.reactionListSize):
        # templist.append(random.choice(id_data_list))
        # id_data_list = templist
        try:
            basecall_list = self.p.map(run_basecall, [k[2] for k in id_data_list])
            id_list = ['>%s_%s\n' % (k[0], k[1]) for k in id_data_list]
            final_str_list = [id_list[i] + basecall_list[i][0] + '\n' for i in xrange(len(id_list))]
            f = open('incomingReads.fa', 'w')
            f.write(''.join(final_str_list))
            f.close()
            current_path = os.getcwd()
            indict = last_batch('incomingReads.fa', current_path, reference_database, args.last_string)
            self.make_decision(indict, read_time)
        except Exception as e:
            logging.info('Exception thrown!')
            print 'EXCEPTION:'
            print e.message

    def reinforce_old_decisions(self, channels):
        """
        here is where we want to check the channels dictionary against our
        old_reject_decision_dict and reinforce any unblocks that we have already
        made a 'skip' decision on
        :param channels: The dictonary of read data from the event sampler
        :return: A dictionary containing the filtered channels
        """
        temp_dict = dict(channels)
        for channel, data in temp_dict.iteritems():
            if channel in self.old_reject_decision_dict.keys():
                if int(data.read_number) == self.old_reject_decision_dict[channel]:
                    self.current_unblock_map[channel] = int(data.read_number)
                    # print 'reinforcing previous skip decision on read ch%s_rd%s and removing it from consideration'
                    # (channel,data.read_number)
                    logging.info(
                        'reinforcing previous skip decision on read ch%s_rd%s and removing it from consideration' % (
                            channel, data.read_number))
                    channels.pop(channel)
            if channel in self.old_sequence_decision_dict.keys():
                if int(data.read_number) == self.old_sequence_decision_dict[channel]:
                    # print 'reinforcing previous sequence decision on read ch%s_rd%s and removing it from
                    # consideration' % (channel,data.read_number)
                    logging.info(
                        'reinforcing previous sequence decision on read ch%s_rd%s and removing it from consideration' %
                        (channel, data.read_number))
                    channels.pop(channel)
        return channels

    def data_received(self, channels):
        """
        This will grab the data as it arrives and throw it into the queue.
        :param channels: The dictionary of channels provided by the event sampler
        """
        # only need this chunk if it is desired to write events from the event sampler each time they are received
        # if len(channels) < 10: idDataList = [(channelName,data.read_number,convert_to_event_array(
        # data)) for channelName,data in channels.iteritems()] else: idDataList = [(channelName,data.read_number,
        # convert_to_event_array(data)) for channelName,data in list(channels.iteritems())[0:10]] self.p.map(
        # create_events_dataset,[(data,channelName,readNumber) for channelName,readNumber,data in idDataList])
        current_time = get_time()
        # logging.info('incoming reads on channels: %s' % channels.keys())
        for channel, data in channels.iteritems():
            logging.info('incoming read: ch%s_rd%s time: %s' % (
                channel, data.read_number, str(time.asctime()).replace(':', '.').replace(' ', '_')))
        if args.skip:
            channels = {k: v for k, v in channels.iteritems() if int(k) % 2 == 0}
        # the following line will further cut down the number of channels that will
        # be analyzed based on whether or not a decision has already been made on a particular read
        # channels = self.reinforce_old_decisions(channels)
        # the following line enforces our thresholds to further minimize the data for basecalling
        channels = enforce_thresholds(channels)
        if len(channels) > 0:
            logging.info('reads to be processed:%s, time: %s' % (
                {channel: data.read_number for channel, data in channels.iteritems()},
                str(time.asctime()).replace(':', '.').replace(' ', '_')))
            input_t = (current_time, channels)
            self.q.put(input_t)

    def manage_queue(self):
        """
        This will run continuously and assign processes to items in the queue.
        We want to make sure the data is still relevant by checking timestamps.
        """
        while True:
            temp_tuple = self.q.get()
            current_time = get_time()
            timing = current_time - temp_tuple[0]
            if timing < self.threshold_time:
                self.process_events(temp_tuple, args.refDB)
            else:
                print 'TIMEOUT'
                channels = temp_tuple[1]
                lost_list = [('ch%s_rd%s' % (channelName, data.read_number))
                             for channelName, data in channels.iteritems()]
                logging.info('TIMEOUT losing reads: ' + str(lost_list))

    def next_unblock_map(self):
        """
        Returns current map of channel_name to read_number that should be
        unblocked but also clears the map in the assumption that the action to
        unblock will be carried out straightaway.
        """
        return self.current_unblock_map.pop_all_and_clear()


class RunningState(object):
    def __init__(self):
        self.keep_running = True

    def closed(self):
        self.keep_running = False


def run_analysis():
    analyser = RUBRIC(args)
    host = args.host
    setup_conditions = {"ignore_first_events": args.ignore_events, "padding_length_events": 0,
                        "events_length": args.consider_events, "repetitions": 1}
    while True:
        state = RunningState()
        try:
            with ReadUntil(host=host,
                           setup_conditions=setup_conditions,
                           data_received=analyser.data_received,
                           connection_closed=state.closed) as my_client:
                # Start sending stuff to our analyser           
                my_client.start()
                print "Client connection started. Beginning unblock loop..."
                logging.info('Connected to server, startup conditions' + str(setup_conditions))
                while state.keep_running:
                    unblock_now = analyser.next_unblock_map()
                    if len(unblock_now) > 0:
                        print "Unblocking channels: ", unblock_now.keys()
                        logging.info("Unblocking channels: %s", unblock_now.keys())
                        my_client.unblock(unblock_now)
                    time.sleep(0.25)
        except socket_error as serr:
            print 'error connecting to server:', serr
            print 'attempting reconnection...'
            time.sleep(1)


def __main__():
    args = get_args()
    fn = '%s.log' % str(time.asctime()).replace(':', '.').replace(' ', '_')
    logging.basicConfig(filename=fn, level=logging.DEBUG)
    model_file = os.path.relpath('nanonet/data/r9.4_template.npy')
    network = np.load(model_file).item()
    run_basecall = partial(nnc.run_basecall, network=network)
    run_analysis()


if __name__ == "__main__":
    __main__()
