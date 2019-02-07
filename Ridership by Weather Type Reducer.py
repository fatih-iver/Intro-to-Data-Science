import sys
import logging

from util import reducer_logfile
logging.basicConfig(filename=reducer_logfile, format='%(message)s',
                    level=logging.INFO, filemode='w')

def reducer():
    '''
    Given the output of the mapper for this assignment, the reducer should
    print one row per weather type, along with the average value of
    ENTRIESn_hourly for that weather type, separated by a tab. You can assume
    that the input to the reducer will be sorted by weather type, such that all
    entries corresponding to a given weather type will be grouped together.

    In order to compute the average value of ENTRIESn_hourly, you'll need to
    keep track of both the total riders per weather type and the number of
    hours with that weather type. That's why we've initialized the variable
    riders and num_hours below. Feel free to use a different data structure in
    your solution, though.

    An example output row might look like this:
    'fog-norain\t1105.32467557'

    Since you are printing the output of your program, printing a debug
    statement will interfere with the operation of the grader. Instead,
    use the logging module, which we've configured to log to a file printed
    when you click "Test Run". For example:
    logging.info("My debugging message")
    Note that, unlike print, logging.info will take only a single argument.
    So logging.info("my message") will work, but logging.info("my","message") will not.
    '''

    riders = 0      # The number of total riders for this key
    num_hours = 0   # The number of hours with this key
    old_key = None

    red = {}
    order = []

    for line in sys.stdin:

        data = line.strip().split('\t')
        this_key, count = data

        if old_key and old_key != this_key:
            print '{0}\t{1}'.format(old_key, num_hours / riders )
            num_hours = 0
            riders = 0

        old_key = this_key
        riders += 1
        num_hours += float(count)
        # your code here
    if old_key:
        print "{0}\t{1}".format(old_key, num_hours / riders )

        """
        key, value = line.strip().split('\t')

        if key in red:
            red[key][0] += 1
            red[key][1] += float(value)
        else:
            red[key] = [1,float(value)]
            order.append(key)
            logging.info(key)
    for key in order:
        print "{0}\t{1}".format(key, red[key][1] / red[key][0])
        """
reducer()
