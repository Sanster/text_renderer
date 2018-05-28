import time


class Timer(object):
    MILLISECOND = 0
    SECOND = 1
    HOUR = 2

    start_time = 0
    ids_start_time = {}

    def __init__(self, show_type=MILLISECOND):
        self.show_type = show_type

    def start(self, tid=None):
        if tid:
            self.ids_start_time[tid] = time.time() * 1000
        else:
            self.start_time = time.time() * 1000

    def end(self, msg, tid=None):
        end_time = time.time() * 1000

        if tid and (tid in self.ids_start_time.keys()):
            total_time = end_time - self.ids_start_time[tid]
        else:
            total_time = end_time - self.start_time

        print_time = total_time
        if self.show_type == Timer.MILLISECOND:
            print_time = total_time
            print_unit = 'ms'
        elif self.show_type == Timer.SECOND:
            print_time = total_time / 1000
            print_unit = 's'
        elif self.show_type == Timer.HOUR:
            print_time = total_time / 1000 / 60 / 60
            print_unit = 'h'

        print("%s: %.3f %s" % (msg, print_time, print_unit))
