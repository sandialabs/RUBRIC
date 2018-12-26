import sys
sys.path.append("..")
from RUBRIC.read_until import ReadUntil
import time
import errno
from socket import error as socket_error
import csv

class MessageStats:
    def __init__(self, pre_encode_time, received_time, total_channel_count, total_event_count):
        self.pre_encode_time = pre_encode_time
        self.received_time = received_time
        self.total_channel_count = total_channel_count
        self.total_event_count = total_event_count
        self.time_diff = received_time - pre_encode_time
        self.events_per_channel = total_event_count / total_channel_count

    def __repr__(self):
        return "MessageStats(pre_encode_time={0}, received_time={1}, total_channel_count={2}, total_event_count={3})".format(
                self.pre_encode_time,
                self.received_time,
                self.total_channel_count,
                self.total_event_count)

class LatencyTestReadUntil(ReadUntil):

    def __init__(self, **kwargs):
        self.message_stats = []
        super(LatencyTestReadUntil, self).__init__(**kwargs)

    def received_server_message(self, msg):
        if super(LatencyTestReadUntil, self).received_server_message(msg):
            print "Messages received: ", len(self.message_stats)
            self.message_stats.append(MessageStats(msg.pre_encode_time,
                                      int(time.time()),
                                      len(msg.channels_update),
                                      sum([len(d.events) for d in msg.channels_update.values()])))


class RunningState:
    def __init__(self):
        self.keep_running=True

    def closed(self, *args):
        self.keep_running=False

def run_latency_test():
    """Runs ReadUntil with particular setup conditions for a given duration,
    then moves onto running another set of setup conditions for a given
    duration. All the time accumulates statistics about each message coming
    back. When the test finishes, when all configurations have been run, the
    statistics are written to csv files ready for analysis/plotting."""
    host = "ws://localhost:9200"

    
    time_and_setup_conditions = [
        (120, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 100, "repetitions": 1}),
        (120, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 200, "repetitions": 1}),
        (120, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 500, "repetitions": 1}),
        (120, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 800, "repetitions": 1}),
        (120, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 1000, "repetitions": 1}),
        (240, {"ignore_first_events": 0, "padding_length_events": 0, "events_length": 1200, "repetitions": 1}),
    ]
    time_and_setup_iter = iter(time_and_setup_conditions)
    total_run_time = sum([x[0] for x in time_and_setup_conditions])

    state=RunningState()
    duration, setup_conditions = time_and_setup_iter.next()
    with LatencyTestReadUntil(host=host,
                              setup_conditions=setup_conditions,
                              connection_closed=state.closed) as my_client:
        # Start sending stuff to our analyser
        my_client.start()
        change_time = time.time() + duration
        print "Client connection started. Will run for {0} seconds".format(total_run_time)
        while state.keep_running:
            time_now = time.time()
            if (time_now > change_time):
                try:
                    duration, setup_conditions = time_and_setup_iter.next()
                except StopIteration:
                    my_client.stop()
                    break
                print "Changing to new conditions:", setup_conditions
                my_client.update_conditions(setup_conditions)
                change_time = time_now + duration
        make_report(my_client.message_stats)

def make_report(ms):
    series={}
    for m in ms:
        series.setdefault(m.events_per_channel, []).append((m.time_diff, m.total_event_count))
    for s, rows in series.items():
        f = open("{0}_events_per_channel.csv".format(s), "ab")
        wr = csv.writer(f, delimiter=' ')
        for row in rows:
            wr.writerow(row)

 
if __name__ == "__main__":
    try:
        run_latency_test()
    except socket_error as serr:
        if serr.errno != errno.ECONNREFUSED:
             raise serr
        print "Server not started?"
