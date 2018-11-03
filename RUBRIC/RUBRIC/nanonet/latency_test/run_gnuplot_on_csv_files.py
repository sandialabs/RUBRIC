import subprocess
import glob
import sys

files=sorted(glob.glob("*_events_per_channel.csv"),
             cmp=lambda x, y: cmp(int(x.split("_")[0]), int(y.split("_")[0]))) 

gnuplot = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)

def plot_args(f):
    return "\"{0}\" using 2:1 title '{0}' with points".format(f)

title = "Time taken in seconds to send different sized messages"
gnuplot.stdin.write("set term dumb 72 40\n")
gnuplot.stdin.write("set xlabel 'message size/events'\n")
gnuplot.stdin.write("set ylabel 'time/s'\n")
gnuplot.stdin.write("set title '{0}'\n".format(title))
gnuplot.stdin.write("plot ")
gnuplot.stdin.write(", \\\n".join([plot_args(f) for f in files]))
gnuplot.stdin.write("\n")
gnuplot.stdin.write("ex\n")
gnuplot.stdin.flush()
