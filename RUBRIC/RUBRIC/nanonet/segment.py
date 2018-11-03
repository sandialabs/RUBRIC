import collections
import numpy as np


__config__ = {
    'mode'                 :'double_abasic',
    'trim_front'           :5,
    'trim_hairpin'         :5,
    'trim_end'             :0,
    'first_n'              :100,
    'min_events'           :200,
    'min_peak_dur'         :0.0,
    'mad_threshold'        :3.5,
    'peak_threshold'       :0.0,
    'min_pt_dur'           :0.15,
    'pt_window'            :0.5,
    'pt_drop'              :0.0,
    'max_pt_search_len'    :0.5,
    'da_min_peak_dur'      :0.001, #0.02
    'da_min_pt_dur'        :0.00, #0.0
    'abasic_range_backup'  :False,
    'use_first_abasic'     :True,
}


def segment(events, section='template', config=__config__):
    """Splitting even data into template and complement sections, returns
    requested section.

    :param events: event data.
    :param config: a configuration parameters object

    """
    assert section in ('template', 'complement'), "section should be template/complement."

    # Locate stall
    med, mad = med_mad(events['mean'][-100:])
    max_thresh = med + 1.48 * 2 + mad
    first_event = locate_stall(events, max_thresh)
    events = events[first_event:]

    data1, data2, results = split_hairpin_abasic(events, config)

    results = {k:results[k] for k in ('start_index_temp', 'end_index_temp', 'start_index_comp', 'end_index_comp')}
    results['first_event'] = first_event

    if section == 'template':
        return data1, results
    else:
        return data2, results


def locate_stall(events, max_threshold, min_events=3):
    """Remove stall section of a read, if present

    :param events: numpy array of events
    :param max_threshold: threshold above which the beginning will be discarded
        e.g. 2 stdvs of the mean model level above median of final 100 events
    :param min_events: minimum number of events we need to see under the threshold before we
        consider the stall to have ended.
    :returns: new sample number at which to start data
    """
    # For when the stall starts below the max_threshold:
    count_above = 0
    start_ev_ind = 0
    for ev_ind, event in enumerate(events[:100]):
        if event['mean'] <= max_threshold:
            count_above = 0
        else:
            count_above += 1

        if count_above == 2:
            start_ev_ind = ev_ind - 1
            break
        
    new_start = 0
    count = 0
    for idx in range(start_ev_ind, len(events)):
        if events['mean'][idx] > max_threshold:
            count = 0
        else:
            count += 1

        if count == min_events:
            # find the time the first event went below - taking just the number
            # away gets the last time *above* the threshold, so add 1
            new_start = idx - min_events + 1
            break

    return new_start


def split_hairpin_abasic(data, parms):
    """ Split hairpins based on abasic in the hairpin.
    :param data: Event data.
    :param parms: Dictionary of parameters to use.

    :returns: Two numpy arrays for the template and complement data. A summary
        is also returned.
    """
    template_data = None
    complement_data = None

    trim_front = parms['trim_front']
    trim_hairpin = parms['trim_hairpin']
    trim_end = parms['trim_end']
    min_events = parms['min_events']
    if parms['mode'] != 'none':
        first_n = parms['first_n']
        peak_threshold = parms['peak_threshold']
        mad_threshold = parms['mad_threshold']
        min_peak_dur = parms['min_peak_dur']
        min_pt_dur = parms['min_pt_dur']
        pt_window = parms['pt_window']
        pt_drop = parms['pt_drop']
        pt_search_len = parms['max_pt_search_len']
        da_min_peak_dur = parms['da_min_peak_dur']
        da_min_pt_dur = parms['da_min_pt_dur']
        abasic_range_backup = parms['abasic_range_backup']
        double = (parms['mode'] == 'double_abasic')
        use_first_abasic = parms['use_first_abasic']

    num_events = data.size
    a, b, c, d = (0, 0, 0, 0)
    la_peak, la_pos, la_dur = (0, -1, 0)
    hp_peak, hp_pos, hp_events, hp_dur = (0, -1, 0, 0)
    pt_level = 0.0
    abasics_found = 0

    # Don't split if too short.
    if num_events < min_events or parms['mode'] == 'none':
        a = trim_front
        b = num_events - trim_end
        if b <= a:
            a = 0
            b = 0
    else:
        # First try trimming front abasic
        trim_gap = trim_front + trim_hairpin
        la_peak, la_pos, la_dur, first_n = find_leader_abasic(data, first_n,
                                                     mad_threshold,
                                                     min_peak_dur,
                                                     trim_gap)
        
        # Then split by specified abasic
        if double:
            # Find if double abasic present
            hp_peak, hp_pos, hp_events, hp_dur, pt_level, abasics_found = \
                split_hairpin_double_abasic(data, first_n, mad_threshold, da_min_peak_dur,
                                            da_min_pt_dur, pt_drop, pt_search_len, use_first_abasic)
        else:
            # Find if single abasic present
            hp_peak, hp_pos, hp_events, hp_dur, pt_level, abasics_found = \
                split_hairpin_single_abasic(data, first_n, mad_threshold, min_peak_dur,
                                     min_pt_dur, pt_window, pt_drop)

        a = trim_front + la_pos if la_pos != -1 else trim_front
        b = hp_pos - trim_hairpin if hp_pos != -1 else num_events - trim_end
        c = hp_pos + hp_events + trim_hairpin if hp_pos != -1 else 0
        d = num_events - trim_end if hp_pos != -1 else 0
    if b > a:
        has_template = True
    else:
        has_template = False
        a = 0
        b = 0
    if d > c:
        has_complement = True
    else:
        has_complement = False
        c = 0
        d = 0
    num_template = b - a
    num_complement = d - c
    dur_template = 0
    dur_complement = 0
    med_level_template = 0
    med_level_complement = 0
    med_sd_template = 0
    med_sd_complement = 0
    range_template = 0
    range_complement = 0
    if has_template:
        template_data = data[a:b]
        dur_template = template_data[-1]['start'] + template_data[-1]['length'] - template_data[0]['start']
        med_level_template, range_template = med_mad(template_data['mean'])
        med_sd_template = np.median(template_data['stdv'])
    if has_complement:
        complement_data = data[c:d]
        dur_complement = complement_data[-1]['start'] + complement_data[-1]['length'] - complement_data[0]['start']
        med_level_complement, range_complement = med_mad(complement_data['mean'])
        med_sd_complement = np.median(complement_data['stdv'])

    if parms['mode'] != 'none' and abasic_range_backup and (not has_template or not has_complement) \
       and not double:
        # Try splitting by range
        new_template, new_complement, new_results = split_hairpin_range(data, parms)
        if new_complement is not None:
            return new_template, new_complement, new_results

    results = {'abasic_index': la_pos,
               'abasic_peak': la_peak,
               'abasic_dur': la_dur,
               'split_index': hp_pos,
               'hairpin_peak': hp_peak,
               'hairpin_dur': hp_dur,
               'hairpin_events': hp_events,
               'hairpin_abasics': abasics_found,
               'pt_level': pt_level,
               'num_events': data.size,
               'num_temp': num_template,
               'num_comp': num_complement,
               'start_index_temp': a,
               'end_index_temp': b,
               'start_index_comp': c,
               'end_index_comp': d,
               'duration_temp': dur_template,
               'duration_comp': dur_complement,
               'median_level_temp': med_level_template,
               'median_level_comp': med_level_complement,
               'median_sd_temp': med_sd_template,
               'median_sd_comp': med_sd_complement,
               'range_temp': range_template,
               'range_comp': range_complement}
    return template_data, complement_data, results


def find_leader_abasic(events, first_n=150, mad_threshold=4.5, min_peak_dur=0, trim_gap=10):
    """Attempt to find a leader abasic at the beginning of a strand. This is performed by taking a
    simple delta-mean of the raw data so that it's independent of normal event detection.

    :param events: numpy record array of events
    :param first_n: number of events to use for finding the leader abasic.
    :param mad_threshold: number of mads above the median the abasic must exceed.
    :param min_peak_dur: minimum duration of the abasic.
    :param trim_gap: minimum trim distance between the leader and hairpin abasics
    :returns: A tuple of:

        * height of the leader abasic peak (or 0 if not found)
        * location of the end of the leader abasic peak (or -1 if not found)
        * duration of the leader abasic peak (or 0 if not found)
        * number of events to start from
        
    :rtype: tuple

    .. note::
        It's important to pass **all** the channel data in, and not just the first few samples, as
        we may not get an accurate measurement of the median current level otherwise.

    """
    la_peak, la_pos, la_dur, la_events, _ = _find_abasic(events, events[:first_n],
                                                         mad_threshold, min_peak_dur)
    
    if (la_pos != -1) and (la_pos + la_events >= first_n - trim_gap):
        first_n = la_pos + la_events + trim_gap + 1
    
    return (la_peak, la_pos + la_events, la_dur, first_n)


def split_hairpin_single_abasic(events, first_n=150, mad_threshold=4.5, min_peak_dur=0,
                                min_pt_dur=0.15, pt_window=0.5, pt_drop=1.5):
    """
    Find the single abasic hairpin
    :param events: numpy record array of events
    :param first_n: number of events to use for finding the abasic
    :param min_peak_dur: minimum duration of the abasic.
    :param min_pt_dur: minimum duration of the pT
    :param pt_window: length of time after hairpin abasic to search for pT
    :param pt_drop: number of mads below the median the pT must be
    :returns: A tuple of:
    
    * height of the leader abasic peak (or 0 if not found)
        * location of the end of the leader abasic peak (or -1 if not found)
        * duration of the leader abasic peak (or 0 if not found)
        * events in the abasic (or 0 if not found)
        * number of events to trim from the front (or left at what was passed in/default)
    
    :rtype: tuple
        
    """

    hp_peak, hp_pos, hp_dur, hp_events, pt_level = _find_abasic(events[first_n:],
                                                                events[first_n:],
                                                                mad_threshold,
                                                                min_peak_dur,
                                                                min_pt_dur,
                                                                pt_window,
                                                                pt_drop)
    abasics_found = 0
    if hp_pos != -1:
        abasics_found = 1

    if hp_pos != -1:
        hp_pos += first_n
        
    return (hp_peak, hp_pos, hp_events, hp_dur, pt_level, abasics_found)


def split_hairpin_double_abasic(events, first_n=150, mad_threshold=4.5, min_peak_dur=0,
                                min_pt_dur=0.15, pt_drop=1.5, max_pt_search_distance=3,
                                use_first_abasic=False):
    """
    Find the double abasic hairpin
    :param events: numpy record array of events
    :param first_n: number of events to use for finding the abasic
    :param min_peak_dur: minimum duration of the abasic
    :param min_pt_dur: minimum duration of the pT
    :param pt_drop: number of mads below the median the pT must be
    :param max_pt_search_distance: maximum time between double abasics
    :returns: A tuple of:
    
        * height of the hairpin abasic peak (or 0 if not found)
        * location of the beginning of the (first) hairpin abasic (or -1 if not found)
        * number of events in the hairpin (or 0 if not found)
        * duration of the hairpin abasic peak(s) (or 0 if not found)
        * level of the pT level (or 0 if not found)
        * number of abasics found
    
    :rtype: tuple
    
    .. note::
        It's important to pass **all** the channel data in, and not just the first few samples, as
        we may not get an accurate measurement of the median current level otherwise.
        
    """
   
    hp_peak, hp_pos, hp_dur, pt_level, hp_events =  _find_hairpin_double_abasic(events[first_n:],
                                                                mad_threshold,
                                                                min_peak_dur,
                                                                min_pt_dur,
                                                                pt_drop,
                                                                max_pt_search_distance,
                                                                use_first_abasic)

    abasics_found = 0
    if hp_pos != -1:
        abasics_found = 2
        
    if hp_pos != -1:
        hp_pos += first_n
                
    return (hp_peak, hp_pos, hp_events, hp_dur, pt_level, abasics_found)


def _find_abasic_candidates(events, mean_threshold, min_peak_dur, leader_peak_height=0,
                            max_events_to_search=None, peak_threshold=0):
    """Finds all the potential abasic candidates meeting the requirements.

    Because this is also used for leader abasic detection, there is the additional functionality to
    continue accruing abasic events once we're past the end of our search window, so that we can be
    sure that we get the entire abasic.

    .. note::
        In the case where there is an abasic at the very end of the events (i.e. events[-1]['mean']
        is greater than :param:`mean_threshold`), the abasic will not be added to the candidates, as
        there is currently no reason we would want this.

    :param np.array events: events to check, possibly bounded by :param:`max_events_to_search`
    :param mean_threshold: minimum threshold event means must exceed to be part of a peak.
    :param min_peak_dur: minimum duration abasic candidates must have.
    :param leader_peak_height: height of the abasic in the leader.
    :param max_events_to_search: the maximum number of events to search for candidates. In the case
        where we're mid-abasic at the end of this window, we'll continue until the abasic ends.
    :param peak_threshold: fraction of :param:`leader_peak_height` candidate abasics must reach (deprecating)
    :returns: abasic peak candidates in the form of (loc, num_events, height, duration).
    :rtype: array of tuples.

    """
    means = events['mean']
    lengths = events['length']
    temp_peak = 0
    temp_loc = 0
    duration = 0
    count = 0
    in_peak = False
    candidates = collections.deque()
    if max_events_to_search is None:
        max_events_to_search = means.size
    for i in xrange(means.size):
        if in_peak:
            if means[i] > mean_threshold:
                count += 1
                duration += lengths[i]
                if means[i] > temp_peak:
                    temp_peak = means[i]
            else:
                in_peak = False
                if duration >= min_peak_dur and temp_peak >= peak_threshold * leader_peak_height:
                    candidates.append((temp_loc, count, temp_peak, duration))
                temp_loc = 0
                temp_peak = 0
                count = 0
                duration = 0
        else:
            if i >= max_events_to_search:
                break
            if means[i] > mean_threshold:
                temp_loc = i
                count = 1
                duration = lengths[i]
                temp_peak = means[i]
                in_peak = True
    return list(candidates)


def _check_for_pT(candidate, events, min_pt_dur, pt_window, pt_max):
    """Check the candidate for a trailing polyT event of the required length.

    :param candidate: candidate tuple as returned by :function:`_find_abasic_candidates()`
    :param min_pt_dur: minimum required duration of polyT.
    :param pt_window: length of time after a candidate to search for polyT.
    :param pt_max: maximum height to be considered a pT.
    :returns: (pt_dur, pt_count, pt_level) tuple if the candidate has polyT, empty tuple otherwise
    :rtype: tuple

    """
    lengths = events['length']
    means = events['mean']

    peak_loc = candidate[0]
    peak_events = candidate[1]

    diff = 0.0
    pos = peak_loc + peak_events
    pt_dur = 0.0
    found = False
    pt_level = pt_max

    # We'll measure the full length of the pT, so we can use that as a tiebreaker.
    while (diff < pt_window or found) and pos < events.size:
        if means[pos] <= pt_max:
            pt_dur += lengths[pos]
            if means[pos] < pt_level:
                pt_level = means[pos]
        else:
            if found:
                break
            else:
                pt_dur = 0.0
        if not found and pt_dur >= min_pt_dur:
            found = True
        diff += lengths[pos]
        pos += 1

    if found:
        pt_events = pos - peak_loc - peak_events
        return pt_dur, pt_events, pt_level
    return ()


def _find_hairpin_double_abasic(events, mad_threshold, min_peak_dur, min_pt_dur,
                                pt_drop, max_pt_search_distance, use_first_abasic=False):
    """Find a double-abasic hairpin. This is slightly complicated by the possibility that the pT may
    or may not be present between the two abasics, due to the strand moving too fast. However, we're
    going to ignore that possibility for the moment, as we haven't got any evidence suggesting that
    it's an issue.

    Abasic candidates are prioritized based on (in order):

        * pT length
        * lower pT height
        * abasic height
        * abasic total length

    .. note::
       Because this is (currently) running in parallel with _find_abasic, and the abasics used in
       the double hairpins are six times as long as the other ones, this should be called with
       min_peak_dur equal to somewhere near six times what it would have been during a normal
       _find_abasic call.

    :param events: numpy record array of events.
    :param mad_threshold: number or mads above the median the abasics must exceed.
    :param min_peak_dur: minimum duration of the abasics.
    :param min_pt_dur: minimum duration of the pT.
    :param pt_drop: number of mads below the median the pT must be.
    :param max_pt_search_distance: the maximum distance between two abasics within which they can be
        considered to be part of a single hairpin.
    :returns:

        * height of the hairpin abasic peak, or 0 if not found.
        * location of the hairpin (or -1 if not found).
        * duration of the hairpin abasic peaks (or 0 if not found). Note that this is the sum of the
          durations of *both* the abasics.
        * level of the pT (or 0 if not found). If we modify detection so that it's possible to have
          a single large abasic which has subsumed the pT then this could theoretically end up as
          zero.
        * number of events in hairpin (start of first abasic to end of second abasic).
    :rtype: tuple

    """
    max_peak = 0
    peak_loc = -1
    peak_dur = 0
    pt_level = 0
    hp_events = 0
    means = events['mean']
    median, mad = med_mad(means)
    mean_threshold = mad_threshold * mad + median

    abasics = _find_abasic_candidates(events, mean_threshold, min_peak_dur)
    candidates = []
    for index, abasic in enumerate(abasics):
        # We'll first check for a second abasic very soon after this one
        # Recall each of these abasics is (abasic_event_index, num_events_in_abasic, ...)
        peak_end = events[abasic[0] + abasic[1]]['start']  # i.e. the start of the next event
        # Add candidates for all cases where the second abasic falls within the search distance
        next_abasic_index = index + 1
        while (next_abasic_index < len(abasics) and
               events[abasics[next_abasic_index][0]]['start'] - peak_end <= max_pt_search_distance):
            candidates.append((abasic, abasics[next_abasic_index]))
            next_abasic_index += 1
        # elif abasic[1] >= 1.5 * min_peak_len:  # Check for an extra-long abasic
        #     candidates.add(('long-abasic', index, abasic))
    if candidates:
        candidates_with_polyT = []
        pt_max = median - 1.4826 * mad * pt_drop  # 1.4826 * mad is approximately 1 stdv.
        for candidate in candidates:
            time_between_abasics = events[candidate[1][0]]['start'] - events[candidate[0][0]
                                                                             + 1]['start']
            pT_test = _check_for_pT(candidate[0], events, min_pt_dur, time_between_abasics, pt_max)
            if pT_test:
                candidates_with_polyT.append((candidate, pT_test))
        if candidates_with_polyT:
            def total_abasic_length(abasic0, abasic1):
                # These are abasics from _find_abasic_candidates()
                # Some slight messiness with abasic1 in case it's at the end of the strand
                return abasic0[3] + abasic1[3]
            # We'll tie break on pT length, inverse pT height, abasic height, total abasic length
            # As a reminder, these tuples are: ((abasic0, abasic1), pt_dur, pt_events, pt_level)
            if use_first_abasic == True:
                best = candidates_with_polyT[0]
            else:
                best = max(candidates_with_polyT,
                           key=lambda x: (x[1][0],
                                         -x[1][2],
                                          max(x[0][0][2],
                                              x[0][1][2]),
                                          total_abasic_length(x[0][0], x[0][1])))

            max_peak = max(best[0][0][2], best[0][1][2])
            peak_loc = best[0][0][0]
            peak_dur = best[0][0][3] + best[0][1][3]
            pt_level = best[1][2]
            hp_events = best[0][1][0] + best[0][1][1] - peak_loc
    return max_peak, peak_loc, peak_dur, pt_level, hp_events


def _find_abasic(all_events, events_to_search, mad_threshold, min_peak_dur,
                 min_pt_dur=None, pt_window=None, pt_drop=None, peak=0.0):
    """Find the best abasic, optionally with the requirement that it possesses a pT.

    :param all_events: numpy record array of all events, for use in threshold calculations.
    :param events_to_search: numpy record array of events to search for an abasic.
    :param mad_threshold: number or mads above the median the abasics must exceed.
    :param min_peak_dur: minimum duration of candidate abasics.
    :param min_pt_dur: minimum duration of the pT.
    :param pt_window: length of time after hairpin abasic to search for the pT.
    :param pt_drop: number of mads below the median the pT must be.
    :param peak: height of the leader abasic.
    :returns:

        * height of the abasic peak, or 0 if not found.
        * location of the beginning of the abasic (or -1 if not found).
        * duration of the abasic peak (or 0 if not found).
        * number of events in the abasic peak or abasic + pT (or 0 if not found).
        * level of the pT (or 0 if not found).
    :rtype: tuple

    """
    max_peak = 0
    peak_loc = -1
    peak_dur = 0
    peak_events = 0
    pt_level = 0
    total_events = 0
    # We compute the med and mad from all the events to give us a consistent threshold for both
    # leader and hairpin abasic detection.
    median, mad = med_mad(all_events['mean'])
    mean_threshold = mad_threshold * mad + median

    candidates = _find_abasic_candidates(all_events, mean_threshold,
                                         min_peak_dur, peak, len(events_to_search))
    if min_pt_dur is None and candidates:  # Not checking for polyT -- just use highest abasic
        # We'll break ties with peak length
        peak_loc, peak_events, max_peak, peak_dur = max(candidates, key=lambda x: (x[2], x[3]))
        total_events = peak_events
    elif candidates:
        candidates_with_polyT = []
        pt_max = median - 1.4826 * mad * pt_drop  # 1.4826 * mad is approximately 1 stdv.
        for candidate in candidates:
            pT_test = _check_for_pT(candidate, events_to_search, min_pt_dur, pt_window, pt_max)
            if pT_test:
                candidates_with_polyT.append((candidate, pT_test))
        if candidates_with_polyT:
            # We'll prioritize based on: abasic height, lower pT height, pT length
            best = max(candidates_with_polyT, key=lambda x: (x[0][2], -x[1][2], x[1][0]))
            peak_loc, peak_events, max_peak, peak_dur = best[0]
            pt_level = best[1][2]
            total_events = peak_events + best[1][1]
        else:
            total_events = peak_events
    return max_peak, peak_loc, peak_dur, total_events, pt_level


def med_mad(data, axis=None):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median
    
    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over 

    :returns: a tuple containing the median and MAD of the data 
    """
    dmed = np.median(data, axis=axis)
    dmad = np.median(abs(data - dmed), axis=axis)
    return dmed, dmad
