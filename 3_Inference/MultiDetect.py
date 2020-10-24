#!/usr/bin/env python3.7
"""

Testbench app for multi-stream, multi-gpu YOLO
File name: MultiDetect.py
Author: Bertel Schmitt, with inspirations from a cast of thousandss
Date last modified: 10/25/2020
Python Version: 3.7
License: This work is licensed under a Creative Commons Attribution 4.0 International License.
"""
from warnings import simplefilter
import os
import sys
from sys import argv
import time
import fcntl
import pickle
import atexit
import threading
from threading import Timer
import multiprocessing
import queue
import ast
import datetime
import subprocess
import tkinter as tk
from tkinter import ttk, Menu, StringVar
import tkinter
import cProfile
from pstats import SortKey
import pstats
import io
import copy
import functools
from timeit import default_timer as timer
from PIL import Image
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import cv2
# for error suppression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Turn off nags for the time being. We'll turn on/off once config file is read
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
#from tensorflow.compat.v1 import logging

#====================================
import gc
from pympler.tracker import SummaryTracker
from pympler import muppy, summary
from pympler import refbrowser
import psutil
import math
# from mem_top import mem_top
#====================================

# Constants
CONST_DASHDASHDASH = '---'

#globals
do_update = True
shutting_down = False


def convert_size(size_bytes, do_comma = False):
    """
    number to mega etc bytesm used in debug
    """

    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    if do_comma:
        return("%s %s" % ('{:,}'.format(s), size_name[i]))
    else:
        return("%s %s" % (s, size_name[i]))


def getappname(ending):
    """
    Return the name of the running app while stripping off ending
    """
    mf = os.path.basename(sys.argv[0])
    if mf.endswith(ending):
        return (mf[:-len(ending)])
    return ("")

def get_parent_dir(n=1):
    """
    Returns the n-th parent dicrectory of the current
    working directory
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")
resource_path = os.path.join(get_parent_dir(1), "3_Inference/MDResource")
data_path = os.path.join(get_parent_dir(1), "Data")
sys.path.append(src_path)
sys.path.append(utils_path)
sys.path.append(resource_path)
pid_file = f'{resource_path}/{getappname(".py")}.pid'
from keras_yolo3.yolo import YOLO  # pylint: disable=C0413
mds = resource_path+"/MultiDetectStatus.py"
if os.path.exists(mds):
    from MultiDetectStatus  import *


simplefilter(action='ignore', category=FutureWarning)

def harakiri():
    '''
    Hack to kill any zombie processes before killing app
    '''
    try:
        mf = os.path.basename(__file__)
    except NameError:
        sys.exit()

    with open(f'{resource_path}/harakiri.sh', 'w+') as file:  # Create a small script
        file.write(f'pkill -f {mf}\n') #kill any instance of running script
        file.write(f'rm -rf {resource_path}/harakiri.sh\n\n') #script performs harakiri on itself
    _ = subprocess.Popen(['/bin/sh', os.path.expanduser(f'{resource_path}/harakiri.sh')])
    time.sleep(1)
    sys.exit()



def resurrect():
    '''
    Hack to kill running process, restart app when all killed
    '''
    mp = __file__
    mf = os.path.basename(__file__)

    with open(f'{resource_path}/resurrect.sh', 'w+') as file:  # Create a small script
        file.write(f'pkill -f  {mf}\n') #kill any instance of running python program
        file.write(f'python {mp}\n') #run the killed python program
        file.write(f'rm -rf {resource_path}/resurrect.sh\n') #delete the script
    _ = subprocess.Popen(['/bin/sh', os.path.expanduser(f'{resource_path}/resurrect.sh')]) #now run the script, kill and restart python program, then delete resurrect.sh



def checkopen(filepath, filearg,auto_open = False, buffering = 0):
    """
    Checks whether file exists, if not, creates file and any necessary directories
    return file handle if autoopen
    """
    if not os.path.isfile(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True) #create directory and any parents if not exist
    if auto_open:
        return(open(filepath, filearg, buffering))
    return()


def get_device_dict():
    """
    Get available GPUs as a dict
    Each GPU returns a dict like {'device': '0', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:00:00.0', 'compute capability': '6.1'}}
    Returned devdict is a dict of dicts, with the GPU index as the key. The whole thing will look something like:
    {'0': {'device': '0', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:21:00.0', 'compute capability': '6.1'},
    '1': {'device': '1', 'name': 'GeForce GTX 1080 Ti', 'pci bus id': '0000:4c:00.0', 'compute capability': '6.1'}}
    Below code is a kludge and could most likely be written in a more elegant way, but it works, and it is called only once a program startup.
    """

    devdict={}
    for i in device_lib.list_local_devices():
        workdict={}
        b  = str(i).split('physical_device_desc:')
        if len(b) > 1 and 'XLA_' not in b[1]:
            subsplit=b[1].replace('"','').replace('\n','').split(', ')
            for it in subsplit:
                pair=it.strip().split(': ')
                if len(pair) > 1:
                    workdict[pair[0].strip()] = pair[1].strip()
            try:
                devdict[workdict['device']] = workdict
            except:
                pass
    return(devdict)

class CTL:
    """
    Container for sundry working variables
    """
    def __init__(self):
        self.procnum = 0

class DefMaster:
    """
    DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
    Defaults for blk_Master, will set mctl in master process.
    Will be overwritten by any existing master settings in config file
    """
    def __init__(self):

        # DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
        # default settings
        self.num_processes = 1 # Number of processes to launch
        self.hush = False #Try to suppress noisy warnings
        self.sync_start = True  #synchronize start of all video streams in an attempt to mitigate time drift between videos
        self.sync_start_wait = 15 #time in seconds to wait for sync_start before timeout. This will be multiplied by num_processes to allow for longer staging due to higher load
        self.all_dead_restart = False #If set, app will restart when all video processes have died. If not set, app will shut down.
        self.master_window_title = "Master" # title for master monitor window
        self.master_window_x_pos = 100    # Where to place the master window
        self.master_window_y_pos = 100    # Where to place the master window
        self.showmaster = True #if true, show master window, if flase, turn master window off
        self.redraw = True #used for optional screen redraw
        self.do_stats =  False        #collect stats on master screen

        #workvars and placeholders
        self.initdict = {}  #dictionary for device stats
        self.statsdict = {} # dictionary for stats messages
        self.devdict = {} # device dictionary
        self.coldict = {} # dictionary of column lengths
        self.gridtop = 0  # workvar for status message
        self.next_etab_statusline = 1
        self.next_ltab_statusline = 4
        self.next_stab_statusline = 4
        self.soundavailable = False #placeholder
        self.masterup = True # placeholder
        self.ost = None #placeholder ost timer
        # DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf

class DefCommon:
    """
    DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf
    Defaults for blk_Common, will set FLAGS for each video process.
    Will be overwritten by any existing per-process setting in config file
    """
    def __init__(self):

        self.testing = False #if True, play video as specified in testvideo. Set true as default in case no video source specified
        self.max_cams_to_probe =  10 #maximun number of usb cameras to probe for. Probing takes time at startup. 10 is a good number
        self.profile = False #if true, profile video_process() for 3000 frames, then show results for each process

        self.testvideo =   resource_path + "/CatVideo.mp4"  #Dummy video file. Default: .../3_Inference/MDResource/testvid.mp4
        self.dingsound =  resource_path + "/chime.mp3"    #Audible alert.  Default: .../3_Inference/MDResource/chime.mp3
        self.silence =     resource_path + "/silence.mp3"  #needed for test of audio capabilities. Default: .../3_Inference/MDResource/silence.mp3
        self.soundalert =  False #play sound when a class specified in presend_trigger  is detected
        self.buffer_depth = 20   #size of frame buffer in frames. Keep low to combat drift, high enough for smooth video. Note: Some webcams do not allow buffers > 10.
        self.avlcam = [] #placeholder for available camera list
        self.workingcam = [] #placeholder for working camera list
        self.do_loop = False           #if set, a video file will loop to beginning when done


        # Families for automatic action. If set as below, "Fluffy" and "Ginger" would be members of the "cat" family, and all "cat" family members would trigger automatic action, such as sound alert (if set and supported) or automatic recording of video (if set)
        # labeldict = {'Fluffy':'cat','Ginger':'cat','Crow':'bird'}
        # presence_trigger = ['cat']

        self.labeldict = {}
        self.presence_trigger = []
        self.ignore_labels =  []  #List of labels for YOLO to ignore and not to report


        self.presence_timer_interval = 20   # seconds until reset
        self.ding_interval = 20             #allow only one ding within that period
        self.monitor_YOLO  = False                 # Experimental!!! Monitor YOLO execution using timers in lieu of proper error reporting. Shut down thread if timed out. Default: False
        self.YOLO_init_AWOL_wait = 20            # Time in seconds to wait for YOLO to successfully initialize. Shutdown if time exceeded
        self.YOLO_detect_AWOL_wait = 20           # Time in seconds to wait for YOLO to come back from detect_image_extended. Shutdown if time exceeded

        self.record_autovideo = False       # If true, automatically record captured video, and store it in output_path
        self.record_with_boxes = False      # If True, record video with boxes (if any ), if False, record without.

        self.output_path = resource_path+ "/videos/"  #basepath for recorded video files. Default: .../TrainYourOwnYOLO/3_Inference/MDResource/videos/ Specify yours in MultiDetect.conf
        self.record_framecode =  False  #record framecode files corresponding to recorded video files
        self.framecode_path =  resource_path+ "/framecode/"  #basepath for framecode files. Default  .../3_Inference/MDResource/framecode/ Specify yours in MultiDetect.conf


        self.maintain_result_log =  False  # Whether to keep a running log with result timings etc.
        self.result_log_basedir =  resource_path+ "/resultlogs/" # Where to keep the result logs. Default  .../TrainYourOwnYOLO/3_Inference/MDResource/resultlogs/ Specify yours in MultiDetect.conf

        self.rolling_average_n =  32      # Length of rolling average used to determine average YOLO-fps


        # To make up for less capable GPUs, or for high frame rates, use these settings to run only every nth frame through YOLO detection
        # For instance, a frame rate of 25fps, and a setting of 2 would result in ~12 frames per second to be run through YOLO, which is well within the capabilities of a moderately-priced GPU
        # These settings also make for considerable power savings

        self.do_only_every_autovideo =  1   #run only every nth image through YOLO during autovideo recording
        self.do_only_every_initial =    1   #run only every nth image through YOLO during normal times
        self.do_only_every_present =    1   #run every nth image through YOLO after activity was detected and until presence_timer_interval times out

        # Default settings for on-screen type. Can be overridden in app
        self.osd =               True       #  True shows on-screen display, False doesn't
        self.osd_fontFace =      0           #  index of cv2.FONT_HERSHEY_SIMPLEX
        self.osd_fontScale =     1.2         #  font size
        self.osd_fontColor =     (0,255, 0)  #  green
        self.osd_fontWeight =    3           #  will be adjusted for smaller/bigger screens
        # global yolo settings. To use different models on a per stream basis, specify these settings in the respective process block
        self.score = 0.45                   #report detections with this confidence score, or higher
        self.run_on_gpu = 0                 #specify gpu to use. "0,1" for both is required by spec, has little or no effect. Set to "-1" to let Keras pick the best GPU
        self.gpu_memory_fraction = 1        #how much GPU memory to claim for process. 1 = 100%, 0.1 = 10% . Performance will suffer when memory-starved, process will crash if GPU memory insufficient
        self.allow_growth = 1                #-1 let Keras decide, 0 disallow, 1 allow memory used on the GPU to dynamically grow.
        # output window settings


        self.video_path = resource_path+ "/CatVideo.mp4" #path to incoming video stream.
        self.model_path =   data_path +  "/Model_Weights/trained_weights_final.h5" # Default location, specify yours in MultiDetect.conf
        self.anchors_path = src_path  + "/keras_yolo3/model_data/yolo_anchors.txt"  # Default location, specify yours in MultiDetect.conf
        self.classes_path = data_path + "/Model_Weights/data_classes.txt" # Default location, specify yours in MultiDetect.conf
        self.iou =   0.9 # intersection over union threshold
        self.show_stats =  False        #If True, cause YOLO object to print stats to console

        self.window_title = "Monitor"              # Title for output monitor window.

        self.window_wide = 1024  # Default width of output window. Specify yours in in the config file
        self.window_high = 600   # Default height of output window. Specify yours in in the config file
        self.window_x = 200      # Default x-position of output window on screen. Set this in config file to move window, also to a separate monitor
        self.window_y = 200      # Default y-position of output window on screen. Set this in config file to move window, also to a separate monitor

        self.presence_counter = 0

        # DEFAULT SETTINGS, DO NOT EDIT!  DEFINE YOUR SETTINGS IN  MultiDetect.conf

class RepeatedTimer(object):
    """
    Calls function every "interval" seconds
    H/T eraoul, https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds
    """

    def __init__(self, interval, function, *args, **kwargs):
      self._timer = None
      self.interval = interval
      self.function = function
      self.args = args
      self.kwargs = kwargs
      self.is_running = False
      self.next_call = time.time()
      self.start()

    def _run(self):
      self.is_running = False
      self.start()
      self.function(*self.args, **self.kwargs)

    def start(self):
      if not self.is_running:
        self.next_call += self.interval
        self._timer = threading.Timer(self.next_call - time.time(), self._run)
        self._timer.start()
        self.is_running = True

    def stop(self):
      self._timer.cancel()
      self.is_running = False


def RetriggerableTimer(*args, **kwargs):
    """
    Global function for Timer
    """
    return _RetriggerableTimer(*args, **kwargs)

class _RetriggerableTimer(object):
    """
    Retriggerable timer
    """
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = Timer(self.interval, self.function)

    def start(self):
        """
        Alias for run
        """
        self.timer.start()

    def reset(self, interval=None):
        """
        Reset the timer (to the old interval if not defined) and restart
        """
        if interval is None:
            interval = self.interval
        self.interval = interval
        self.timer.cancel()
        self.timer = Timer(self.interval, self.function)
        self.timer.start()

    def cancel(self):
        """stop the timer"""
        self.timer.cancel()

    def is_alive(self):
        """signal running, or not"""
        return(self.timer.is_alive())

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    Routine courtesy https://stackoverflow.com/users/772487/jeremiahbuddha

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def listcams(maxcams = 10):
    """
    List available and working cams. Routine courtesy https://stackoverflow.com/users/2132157/g-m
    """
    dev_port = 0
    working_ports = []
    available_ports = []
    for _ in range(0,maxcams):
        camera = cv2.VideoCapture(dev_port) # pylint: disable=E1101
        if camera.isOpened():
            is_reading, _ = camera.read()
            if is_reading:
                working_ports.append(dev_port)
            else:
                available_ports.append(dev_port)
        dev_port +=1
    return(available_ports,working_ports)


def settext(window, mytext, myrow,  mycol,  font = "Courier", bold = False, size = 12, anchor = "SW", columnspan = 26, borderwidth = 0):
    """
    draw text on a tk window
    """
    mylabel = None
    thefont = f"{font} {size}"
    if bold:
        thefont = thefont + " bold"
    try:
        mylabel = tk.Label(window, text = mytext , font = thefont, borderwidth = borderwidth)
        mylabel.grid(sticky=anchor, row = myrow, column = mycol, columnspan = columnspan)
    except:   #tkinter does not return proper error
        pass
    return(mylabel)

def setrows(window, rows):
    """Create empty rows"""
    for n in range(0,rows+1):
        settext(window,"   ",n,0,columnspan=1)
    return()

def setrows3col(window, rows):
    """Create empty rows"""
    for n in range(0,rows+1):
        settext(window,"   ",n,0,columnspan=1)
        settext(window,' ' *50,n,1,columnspan=1)
        settext(window,' ' *15,n,2,columnspan=1)
        settext(window,' ' *15,n,3,columnspan=1)


    return()

def clear_tk_window(window):
    """
    Wipe a tkinter window clean
    """
    for widget in window.winfo_children():
        widget.destroy()

def button_record_all(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,Command = "startrecording")

def button_stoprecord_all(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,Command = "stoprecording")

def button_audio_on(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,Command = "audioon")

def button_audio_off(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0, Command = "audiooff")

def button_autovid_on(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0, Command = "autovidon")

def button_autovid_off(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0, Command = "autovidoff")

def button_osd_on(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "osdon")

def button_osd_off(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "osdoff")

def button_hush_on(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "hushon")

def button_hush_off(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "hushoff")

def button_stats_on(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        writethelog("Stats ON",my_mctl.procnum)
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "statson")
        my_mctl.do_stats = True
        return(my_mctl)

def button_stats_off(my_mctl, doqueue = True):
    """Button action routine"""
    if doqueue:
        writethelog("Stats OFF",my_mctl.procnum)
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0,   Command = "statsoff")
        my_mctl.do_stats = False
        return(my_mctl)

def button_window_redraw(my_mctl, doqueue = True):
    """Cause video_processes to set their output windows into their proper places"""
    if doqueue:
        my_mctl = put_in_queue(my_mctl, From = -1 ,To = 0, Command = "windowredraw")

def button_shutdown(my_mctl):
    """Process shutdown"""
    global shutting_down
    shutting_down = True
    my_mctl.shutdown_action = -1 #shutdown
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "abortvideowriter", 'Args':{}}, my_mctl.procdict)
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "shutdown", 'Args':{}}, my_mctl.procdict)
    save_app_status(my_mctl)
    panic_shutdown_timer = RetriggerableTimer(30, harakiri) # timer for emergency shutdown in case not all processes acknowledge shutdown command
    panic_shutdown_timer.start()
    #master_shutdown(my_mctl)
    return(my_mctl)

def button_resurrect(my_mctl):
    """Process shutdown"""
    global shutting_down
    shutting_down = True
    save_app_status(my_mctl)
    my_mctl.shutdown_action = 1 #resurrect
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "abortvideowriter", 'Args':{}}, my_mctl.procdict)
    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "shutdown", 'Args':{}}, my_mctl.procdict)
    panic_shutdown_timer = RetriggerableTimer(30, resurrect) # timer for emergency shutdown in case not all processes acknowledge shutdown command
    panic_shutdown_timer.start()
    return(my_mctl)


def save_app_status(the_mctl):
    '''Save master window position and enay other app data'''
    thefile = resource_path+"/MultiDetectStatus.py"
    geo = the_mctl.window.geometry()
    if geo == None or geo == '' or geo == '1x1+0+0':
        return
    with open(thefile, "w+") as f:
        f.write(f"my_geometry = '{the_mctl.window.geometry()}'"+"\n")
    return()



def writeobj(my_mctl):
    gc.collect()
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    sums=sorted(sum1, key=lambda x: x[2],reverse = True)

    nfile = argv[0]
    nfile = nfile[:-3]+f"_objd_{my_mctl.procnum}" + ".log"
    process = psutil.Process(os.getpid())

    with open(nfile, 'a+') as f:

        basehead= "\n" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if my_mctl.do_stats:
            basehead = basehead+ " Stats ON"

        f.write(basehead+"\n")

        f.write(f"All objects: {len(all_objects)}"+"\n")
        totalmem=process.memory_info().rss

        f.write(f"Process mem: {'{:,}'.format(totalmem)} - {convert_size(totalmem, do_comma = True)}"+"\n" )


        for number, value in enumerate(sums, start=1):
            f.write(f"{str(number).rjust(3)} {value}" +  "\n")
            if number >= 100:
                break




def do_masterscreen(my_mctl):
    """
    Monitor screen - set-up master screen
    """
    global showmaster
    if not showmaster:
        return()
    setrows(my_mctl.stab,my_mctl.totallines)
    line_1 = "GPU Stats " + ' ' *  70
    #.ljust(my_mctl.maxline).rjust(my_mctl.maxline)
    line_2 = ("GPU#".rjust(4) +   "  GPU Name ".ljust(25) + "   CCap".rjust(5)  + "            ")
    #.ljust(my_mctl.maxline).rjust(my_mctl.maxline)
    line_5 = "Current Stats"
    # "                                                  "
    #.ljust(my_mctl.maxline).rjust(my_mctl.maxline)
    line_6 = f'{"Proc".rjust(4)}{"Frames".rjust(9)}{"FDiff".rjust(7)}{"Seconds".rjust(9)}{"SecDiff".rjust(8)}{"IFPS".rjust(5)}{"EFPS".rjust(5)}{"YFPS".rjust(5)}{"OFPS".rjust(5)}{"n".rjust(3)}{"Timestamp".rjust(12)}{"GPU".rjust(4)}{"FRAC".rjust(6)}'
    #.ljust(my_mctl.maxline).rjust(my_mctl.maxline)
    settext(my_mctl.stab,line_1,1,1, bold = True, size = 16, font = "Arial")
    settext(my_mctl.stab,line_2,2,1, bold = True)

    mybase = 2

    # draw GPU info
    for gpunum in my_mctl.devdict:
        mybase = mybase + 1
        locdict = my_mctl.devdict[gpunum]
        l = (gpunum.rjust(4) + "  " +locdict['name'].ljust(25) + locdict['compute capability'].rjust(5))
        #.ljust(my_mctl.maxline).rjust(my_mctl.maxline)
        settext(my_mctl.stab, l, mybase,  1)
    mybase = mybase + 2
    settext(my_mctl.stab,line_5,mybase,1, bold = True, size = 16, font = "Arial")
    button_son= tk.Button(my_mctl.stab, text="Stats On",  width =5, font = "Arial 9", command=functools.partial(button_stats_on, my_mctl))
    button_son.grid(row=mybase, column=7, ipadx = 5 , sticky='SW')
    button_soff= tk.Button(my_mctl.stab, text="Stats Off",  width =5,font = "Arial 9", command=functools.partial(button_stats_off, my_mctl))
    button_soff.grid(row=mybase, column=8, ipadx = 5 , sticky='SW')

    mybase = mybase + 1
    settext(my_mctl.stab,line_6,mybase,1, bold = True)
    my_mctl.topdataline = mybase + 1  # set top location for ongoing data lines
    my_mctl.window.attributes("-topmost", True)
    try:
        my_mctl.stab.update()
        my_mctl.window.update()
    except:  # tkinter does not return proper error type
        pass

    return (my_mctl)


def get_currentline(my_mctl, procnum):
    """
    Monitor screen - format current data for each process
    """
    mpos = int(my_mctl.statsdict[str(procnum)]['POS_FRAMES'])
    msec = my_mctl.statsdict[str(procnum)]['POS_MSEC']
    ofps = my_mctl.statsdict[str(procnum)]['StrFPS']
    mstamp = my_mctl.statsdict[str(procnum)]['TimeStamp']
    yfps = my_mctl.statsdict[str(procnum)]['MaxYoloFPS']
    n = int(my_mctl.statsdict[str(procnum)]['DoOnlyEvery'])
    infps = my_mctl.statsdict[str(procnum)]['InFPS']
    mygpu= str(my_mctl.initdict[str(procnum)]['run_on_gpu'])
    is_looping = my_mctl.statsdict[str(procnum)]['Looping']
    try:
        memfrac = '{:.2f}'.format(round(my_mctl.initdict[str(procnum)]['gpu_memory_fraction'], 2))
    except(ZeroDivisionError, TypeError) as e:
        memfrac = CONST_DASHDASHDASH

    if infps > 1000 or infps < 1 : #bogus fps
        strefps = strinfps = stryfps = CONST_DASHDASHDASH
    else:
        strinfps= '{:.1f}'.format(round(infps, 1))

        if yfps > 0:
            strefps = '{:.1f}'.format(round(infps/n, 1))
            stryfps = '{:.1f}'.format(round(yfps, 1))
        else:
            strefps = stryfps = CONST_DASHDASHDASH


    if msec/1000 <= 1: #reports no proper msec
        msecds = strmsec = CONST_DASHDASHDASH
    else:
        try:
            msecds  = '{:.0f}'.format(round(msec / 1000 - int(my_mctl.statsdict['1']['POS_MSEC']) / 1000, 0))
        except KeyError:
            msecds = CONST_DASHDASHDASH

        strmsec = '{:.0f}'.format(round(msec / 1000, 0))

        # no diff for #1, compare others with #1
    if procnum == '1' or is_looping:
        mposds = msecds = CONST_DASHDASHDASH
    else:
        try:
            mposds = str(int(mpos - int(my_mctl.statsdict['1']['POS_FRAMES'])))
        except KeyError:
            mposds = CONST_DASHDASHDASH


    if mpos <= 0:  # bogus, or unavailable frame position
        mpos = mposds = CONST_DASHDASHDASH  # Frame position should be > 0, and in that case, frame difference would not make sense either
    return(f"{str(procnum).rjust(4)}{str(mpos).rjust(9)}{mposds.rjust(7)}{strmsec.rjust(9)}{msecds.rjust(8)}{strinfps.rjust(5)}{strefps.rjust(5)}{stryfps.rjust(5)}{'{:.1f}'.format(round(ofps, 1)).rjust(5)}{str(n).rjust(3)}{milliconv(mstamp).rjust(12)}{mygpu.rjust(4)}{memfrac.rjust(6)}")



def do_dataline(the_mctl, currproc, emerg = ""):
    """
    Monitor screen - draw data line for a process
    """
    global showmaster
    if not showmaster:
        return
    if emerg == "":
        mytext =  f'{get_currentline(the_mctl, str(currproc))}'
        #.ljust(the_mctl.maxline).rjust(the_mctl.maxline)
    else:
        mytext = (f'{str(currproc).rjust(4)}   {emerg}'+ ' ' * the_mctl.maxline)

    #ljust(the_mctl.maxline).rjust(the_mctl.maxline)

    if currproc not in the_mctl.datalinedict:
        s = StringVar()
        slabel = ttk.Label(the_mctl.stab, textvariable = s, font=("Courier", 12)).grid(sticky='SW', row = the_mctl.topdataline + int(currproc) -1 , column = 1, columnspan = 200)  #, padx = 30, pady = 0
        s.set(mytext)
        the_mctl.datalinedict[str(currproc)] = [slabel, s]
    else:
        myrec = the_mctl.datalinedict.get(str(currproc),None) # get a prior text label
        myrec[1].set(mytext)

    return (the_mctl)

def mk_statuslabel(the_mctl, myproc, mytext, mycol):
    '''Create statuslabel'''
    ss = StringVar()
    slabel= ttk.Label(the_mctl.ltab, textvariable = ss, font=("Courier", 12)).grid(sticky='SW', row = the_mctl.gridtop+3+myproc , column = mycol, columnspan = 26)
    ss.set(mytext)
    return([slabel, ss])

def statuslabel(the_mctl, myproc, mytext, mycol):
    '''Make a statuslabel, and file it in the statuslinedict.'''
    if the_mctl.statuslinedict.get(str(myproc),None) == None: #This proc not on file
        the_mctl.statuslinedict[str(myproc)] = {} # create empty record
    if the_mctl.statuslinedict.get(str(myproc), None).get(f'col{mycol}',None) == None: #this label not on file
        the_mctl.statuslinedict[str(myproc)][f'col{mycol}'] = mk_statuslabel(the_mctl, myproc, mytext, mycol)
    else:
        curlabel = the_mctl.statuslinedict.get(str(myproc),None).get(f'col{mycol}',None)
        if curlabel == None:
            raise KeyError(f'KeyError @ line {sys._getframe().f_lineno -2}, statuslabel logic failed. Proc/line: {myproc}, Text: {mytext}, Column: {mycol} ') # pylint: disable=W021
            sys.exit()
        curlabel[1].set(mytext)
    return(the_mctl)

def do_statusline(my_mctl, my_myproc, themsg):
    """
    display a status line
    """
    assert(my_mctl != None)
    my_iam = str(my_myproc)
    padl = [' ' * 50, ' ' * 15, ' ' * 15] # list of pad characers
    if themsg == [] :
        return()
    if themsg[0] != '': #First col has msg
        mymsg =(f"{str(my_iam).rjust(3)}: {themsg[0]}" +  ' ' * 50)[:50]
        try:
            my_mctl=statuslabel(my_mctl, my_myproc, mymsg,1)
        except:
            pass
    if themsg[1] != '': #2nd col has msgs
        mymsg = (themsg[1] +  ' ' * 15)[:15]
        my_mctl=statuslabel(my_mctl, my_myproc, mymsg,2)
    if themsg[2] != '': #3rd col has msgs
        mymsg = (themsg[2] +  ' ' * 20)[:20]
        my_mctl=statuslabel(my_mctl, my_myproc, mymsg,3)
    try:
        my_mctl.ltab.update_idletasks()
    except:  # tkinter does not return proper error type
        pass
    return()

def testparam(mFLAGS):
    """
    Validate parameters for init_YOLO. Return True for all O.K., False and reasonm for not ok
    """
    if not hasattr(mFLAGS,'model_path'):
        return(False,f"must specify model_path","no_YOLO")
    if hasattr(mFLAGS,'model_path') and not os.path.isfile(mFLAGS.model_path):
        return(False,f"model_path incorrect","no_YOLO")
    if not hasattr(mFLAGS,'anchors_path'):
        return(False,f"must specify anchors_path","no_YOLO")
    if hasattr(mFLAGS,'anchors_path') and not os.path.isfile(mFLAGS.anchors_path):
        return(False,f"anchors_path incorrect","no_YOLO")
    if not hasattr(mFLAGS,'classes_path'):
        return(False,f"must specify classes_path","no_YOLO")
    if hasattr(mFLAGS,'classes_path') and not os.path.isfile(mFLAGS.classes_path):
        return(False,f"classes_path incorrect","no_YOLO")
    if hasattr(mFLAGS,'iou') and not  0 < mFLAGS.iou <= 1:
        return(False, f"iou {mFLAGS.iou} out of range","stop")
    if hasattr(mFLAGS,'gpu_memory_fraction') and not 0 < mFLAGS.gpu_memory_fraction <= 1:
        return(False, f"gpu_memory_fraction {mFLAGS.gpu_memory_fraction} out of range","stop")
    if hasattr(mFLAGS,'allow_growth') and not -1 <= mFLAGS.allow_growth <= 1:
        return(False, f"allow_growth {mFLAGS.allow_growth} out of range","stop")
    if hasattr(mFLAGS,'ignore_labels') and type(mFLAGS.ignore_labels) != list:
        return(False, f"ignore_label {mFLAGS.ignore_labels} not a list","stop")

    return(True,"","proceed")



def init_yolo(FLAGS):
    """
    create a yolo session
    Removed "gpu_num" to avoid confusion. gpu_num allegedly allows running THE SAME SESSION on multiple GPUs, but in my testing, it didn't do much, if anything.
    If you need the parameter, simply add   "gpu_num": FLAGS.gpu_num,   to the parameter block below, and add "gpu_num:  1" (or 2, or 3 , or ... without the quotes)  in the config file
    This uses a slightly changed Yolo. In the old version, detect_image returns out_prediction, image. The new version of detect image, detect_image_extendedm, returns image, labels, elapsed_time, out_prediction_extended
    Most of the settings can be set in the config file
    """
    if FLAGS.hush:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)
        silence(True)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        tf.logging.set_verbosity(tf.logging.WARN)
        silence(False)


    mykwargs = {
        "model_path": FLAGS.model_path,
        "anchors_path": FLAGS.anchors_path,
        "classes_path": FLAGS.classes_path,
        "score": FLAGS.score,
        "model_image_size": (416, 416),
        "hush": FLAGS.hush,
    }

    # only set optional parameters that have been specified, allow defaults to be set for all others

    if hasattr(FLAGS, 'gpu_num'):
        mykwargs['gpu_num'] = FLAGS.gpu_num
    if hasattr(FLAGS, 'iou'):
        mykwargs['iou'] = FLAGS.iou
    if hasattr(FLAGS, 'run_on_gpu'):
        mykwargs['run_on_gpu'] = FLAGS.run_on_gpu
    if hasattr(FLAGS, 'gpu_memory_fraction'):
        mykwargs['gpu_memory_fraction'] = FLAGS.gpu_memory_fraction
    if hasattr(FLAGS, 'allow_growth'):
        mykwargs['allow_growth'] = FLAGS.allow_growth
    if hasattr(FLAGS, 'ignore_labels'):
        mykwargs['ignore_labels'] = FLAGS.ignore_labels

    yolo = YOLO( **mykwargs )
    return (yolo)

def print_yolo_args(FLAGS):
    print(f"model_path: {FLAGS.model_path} \nanchors_path: {FLAGS.anchors_path}\nclasses_path: {FLAGS.classes_path}\nscore: {FLAGS.score}\nhush: {FLAGS.hush}\niou : {FLAGS.iou}\nrun_on_gpu : {FLAGS.run_on_gpu}\ngpu_memory_fraction : {FLAGS.gpu_memory_fraction}\nallow_growth : {FLAGS.allow_growth}\nignore_labels : {FLAGS.ignore_labels}\n")


def writethelog(x, procid=""):
    """
    Write x to log
    """
    nfile = argv[0]
    nfile = nfile[:-3]+f"_{procid}" + ".log"
    with open(nfile, 'a+') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "  " + str(x) + "\n")
    return ()

def printandfile(x, procid=""):
    """
    Print x to console and write to log
    """
    print(x)
    writethelog(x, procid)
    return(0)

def yolo_timeout(kwargs):
    """
    Routine to run when YOLO did not return from init
    """
    myctl = kwargs.pop('myctl') #assign , and remove
    kwargs['number'] = myctl.procnum
    writethelog(f"Proc {myctl.procnum} YOLO_timeout exceeded, what: {kwargs.get('what')} where: {kwargs.get('where')} why: {kwargs.get('why')}. Shutting down ")
    #make commandset by hand
    my_commandset = {'To': -1, 'From': myctl.procnum , 'Command': "procdied" , 'Args': kwargs}
    myctl.out_queue.put_nowait(my_commandset)
    time.sleep(1)

    my_commandset = {'To': -1, 'From': myctl.procnum , 'Command': 'error' , 'Args': {'ErrMsg': 'YOLO did not return' ,'Action': "Shutting down"}}
    myctl.out_queue.put_nowait(my_commandset)
    #my_commandset = {'To': -1, 'From': myctl.procnum , 'Command': 'status' , 'Args': {'StatusMsg': [mymsg, '','Shutdown']}}
    #myctl.out_queue.put_nowait(my_commandset)
    time.sleep(2)
    vidproc_shutdown(myctl, echo = False, what = kwargs.get('what'), where = kwargs.get('where'),  why = kwargs.get('why'))
    return ()

def all_running_timeout(*args):
    mymctl = args[0]
    setattr(mymctl, 'all_running', True)
    return(mymctl)

def presence_timeout():
    """
    Actions when presence_timer times out
    """
    return ()

def ding_timeout():
    """
    Audible prompt window timed out. No action needed
    """
    return ()

def soundtest(FLAGS):
    """
    Test audio with FLAGS.silence. Return True if successful, false if not.
    """
    try:
        sound = AudioSegment.from_mp3(FLAGS.silence)
    except FileNotFoundError:
        return(False)
    else:
        return(dosound(sound,FLAGS.hush))

def playsound(FLAGS):
    """
    Play sound at sound file. Return True if successful, false if not. Needs working sound system. If sound not working.
    app will not play sound now, and won't try playing sound later
    """
    try:
        sound = AudioSegment.from_mp3(FLAGS.dingsound)
    except FileNotFoundError:
        return(False)
    else:
        return(dosound(sound,FLAGS.hush))

def dosound(sound,hush):
    """
    play sound, without screen clutter if hush is set
    """
    if hush:
        with suppress_stdout_stderr():
            play(sound)
    else:
        play(sound)
    return(True)


def vidproc_shutdown(myctl, echo = False, what = '', where =' ', why = ''):
    """
    Shut down this video_process. If echo == True, signal other video_process(es) to do same
    """
    writethelog("vidproc_shutdown: queueing Process shut down")
    if what == '':
        what = 'Process shut down'
    myctl = put_in_queue(myctl, From = myctl.procnum, To=-1, Command='status', Args={'StatusMsg': [what,where,why]})
    time.sleep(1)
    writethelog(f"Process {myctl.procnum} shutdown, {what} {where} {why}")
    if echo:
        print("vidproc_shutdown: echo-queueing stoprecording, shutdown")

        myctl = put_in_queue(myctl, From = myctl.procnum,To = 0,Command = "stoprecording")
        myctl = put_in_queue(myctl, From = myctl.procnum,To = 0,Command = "shutdown")

    writethelog("vidproc_shutdown: queueing procdied")
    if why != '':
        myctl = put_in_queue(myctl, From = myctl.procnum , To = -1, Command="procdied", Args={'number': myctl.procnum , 'reason': why})
    else:
        myctl = put_in_queue(myctl, From = myctl.procnum , To = -1, Command="procdied", Args={'number': myctl.procnum})

    writethelog("vidproc_shutdown: shutdown action")
    try:
        myctl.cv2.destroyWindow(myctl.window_title)
    except(AttributeError,cv2.error):
        pass

    if myctl.out is not None and myctl.out.isOpened():
        myctl.out.release()
        myctl.cv2.VideoWriter = None
    try:
        myctl.ding_timer.cancel()
    except AttributeError:
        pass
    try:
        myctl.presence_timer.cancel()
    except AttributeError:
        pass
    try:
        myctl.yolo.close_session()
        del myctl.yolo
    except AttributeError:
        pass
    try:
        myctl.vid.release()
    except AttributeError:
        pass
    try:
        myctl.out.release()
    except AttributeError:
        pass
    try:
        myctl.fcf.close()
    except AttributeError:
        pass
    try:
        myctl.rsf.close()
    except AttributeError:
        pass
    raise SystemExit(f"Shutting down video_process #{myctl.procnum}")


def startup_shutdown():
    """ registered exit routine of startup"""
    writethelog("Startup - shutdown")
    sys.exit()


def master_shutdown(my_mctl, why = '', restart = False):
    """
    Shutdown all video-process(es), followed by master_process
    """
    global do_update, showmaster
    save_app_status(my_mctl)
    showmaster = False
    do_update = False
    if restart:
        printandfile(f"Initiating master restart - {why}")
    else:
        printandfile(f"Initiating master shutdown - {why}")

    send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "abortvideowriter", 'Args':{}}, my_mctl.procdict)
    time.sleep(2)
    # prevent further status updates
    try:
        my_mctl.rt.stop() # stop window update timer
    except AttributeError:
        pass  # timer already gone

    # signal all child propcesses to shutdown.
    my_mctl = put_in_queue(my_mctl,  From = -1, To=0, Command= 'shutdown', Args={})

    try:
        my_mctl.window.destroy()
    except:
        pass
    #kill off all child processes
    main_killer(my_mctl)
    #raise SystemExit("Shutting down Master ...")
    if restart:
        resurrect()
    else:
        harakiri() # do just that
    sys.exit()


def start_recording(myctl, putqueue=False):
    """
    Sundry stuff to do before recording
    """
    myctl.do_only_every = myctl.do_only_every_autovideo
    basenamevid=f'{datetime.datetime.now().strftime("%y%m%d")}/P{myctl.procnum}/{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}_P{myctl.procnum}.mp4'
    if myctl.output_path.strip().endswith("/"):
        myctl.videofile=myctl.output_path.strip()+basenamevid
    else:
        myctl.videofile=myctl.output_path.strip()+"/"+basenamevid

    if myctl.framecode_path.strip().endswith("/"):
        myctl.fcfile=myctl.framecode_path.strip()+basenamevid+".fc"
    else:
        myctl.fcfile=myctl.framecode_path.strip()+"/"+basenamevid+".fc"
    myctl.is_recording = True
    myctl.framecounter = 0
    if putqueue:
        myctl = put_in_queue(myctl, From = myctl.procnum, To=0, Command="startrecording", Args={})
    return (myctl)


def stop_recording(myctl, putqueue=False, delete = False):
    """
    switch off recorder via
    myctl properties, signal all to do same if putqueue True
    """
    myctl.do_only_every = myctl.do_only_every_initial
    myctl.is_recording = False
    myctl.autovideo_running = False
    if delete:
        myctl.presence_counter = -1 # signal deletion to fileops
    if putqueue:
        myctl = put_in_queue(myctl, From = myctl.procnum, To=0, Command="stoprecording", Args={'Delete': delete})
    return (myctl)

def preroll(myctl, frames):
    """roll some video frames"""
    for _ in range(0, frames):
        _, _ = myctl.vid.read()
    return ()

def milliconv(ts):
    """convert float timestamp to hr:min:sec.1"""
    _ , mytime = datetime.datetime.fromtimestamp(ts).strftime(
        '%Y-%m-%d %H:%M:%S.%f').split()  # convert to date and time
    t, rm = mytime.split(".")  # split off the remaining millisecs
    rmf = round(float(f"0.{rm}"), 1)  # round remainder to 1 decimal
    ntime = (f"{t}.{str(int(rmf * 10))[0]}")  # stick it to time removing 0, and making sure only 1 char
    return (ntime)

#def getoflfps(mycv2, vid, init_frames, init_msec):
def getoflfps(myctl, init_frames, init_msec):

    """Try getting official fps, either by direct query, or by calculation
    return fps, mode 0 = CAP_PROP_FPS, 1 = POS_FRAMES/POS/MSEC, 2 = timed loop"""
    oflfps = int(round(myctl.vid.get(myctl.cv2.CAP_PROP_FPS), 0))
    if 0 < oflfps < 200:
        return (oflfps, 0)  #looks like OK fps, set mode 0

    # We need a few seconds of frame history for POS_FRAMES/POS_MSEC to work anyway, so we do a preroll, and time it here, in case POS_FRAMES/POS_MSEC also fails
    start = time.time()
    num_frames = 24
    myctl = put_in_queue(myctl, From = myctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['','','Preroll']})
    preroll(myctl,num_frames)
    seconds = (time.time() - start)
    loopfps = round(num_frames/seconds,1)
    if ('CAP_PROP_POS_MSEC' in supplist and 'CAP_PROP_POS_FRAMES' in supplist):  # try POS_FRAMES/POS_MSEC substitute
        oflfps = round((myctl.vid.get(myctl.cv2.CAP_PROP_POS_FRAMES) - init_frames) / (myctl.vid.get(myctl.cv2.CAP_PROP_POS_MSEC) - init_msec) * 1000, 1)
        if 0 < oflfps < 200: #sanity check
            return(oflfps, 1)  #looks like OK fps, set mode 1
    # all getprops methods failed, so return brute force timing loop
    return(loopfps,2)



def list_supported_capture_properties(cap, mycv2):
    """
    List the properties supported by the capture device.
    """
    print("")
    print("===============================================================================")
    print("Test of supported capture properties. Ignore errors")
    supported = list()
    for attr in dir(mycv2):
        if attr.startswith('CAP_PROP'):
            try:
                if cap.get(getattr(mycv2, attr)) != -1:
                    supported.append(attr)
            except:
                pass
    print("End of test")
    print("===============================================================================")
    print("To turn off nagging messages, set hush to True in MultiDetect.conf")
    print("===============================================================================")
    print("")
    return supported

def makectl(FLAGS):
    """
    set up a subset of FLAGS as a multipurpose-object
    """
    myctl = CTL()  # make new controller
    setattr(myctl, 'procnum', FLAGS.procnum)
    setattr(myctl, 'record_autovideo', bool(FLAGS.record_autovideo))
    setattr(myctl, 'presence_timer_interval', FLAGS.presence_timer_interval)
    setattr(myctl, 'presence_counter', 0)
    setattr(myctl, 'autovideo_running', False)
    setattr(myctl, 'do_only_every_initial', FLAGS.do_only_every_initial)
    setattr(myctl, 'do_only_every_autovideo', FLAGS.do_only_every_autovideo)
    setattr(myctl, 'video_path', FLAGS.video_path)
    setattr(myctl, 'testvideo', FLAGS.testvideo)
    setattr(myctl, 'testing', FLAGS.testing)
    setattr(myctl, 'output_path', FLAGS.output_path)
    setattr(myctl, 'is_recording', False)
    setattr(myctl, 'is_test', False)
    setattr(myctl, 'run_on_gpu', FLAGS.run_on_gpu)
    setattr(myctl, 'gpu_memory_fraction', FLAGS.gpu_memory_fraction)
    setattr(myctl, 'osd_help', False)
    setattr(myctl, 'osd', FLAGS.osd)
    setattr(myctl, 'fontFace', FLAGS.osd_fontFace)
    setattr(myctl, 'fontScale', FLAGS.osd_fontScale)
    setattr(myctl, 'fontColor', FLAGS.osd_fontColor)
    setattr(myctl, 'fontThickness', int(FLAGS.osd_fontWeight))
    setattr(myctl, 'window_title', FLAGS.window_title)
    setattr(myctl, 'framecode_path', FLAGS.framecode_path)
    setattr(myctl, 'framecounter', 0)
    setattr(myctl, 'videofile', "")
    setattr(myctl, 'fcfile', "")
    setattr(myctl, 'out', None) #placeholder
    setattr(myctl, 'fcf', None) #placeholder
    setattr(myctl, 'video_fps', 0) #placeholder
    setattr(myctl, 'do_stats', FLAGS.do_stats) #placeholder

    return (myctl)


def vidproc_checkqueue(myctl, FLAGS):
    """
    Verify content of queue coming into the video process. Operate on results if needed
    """
    while not myctl.in_queue.empty():
        # work the queue
        try:
            # commandset structure {'To':process (1....),master (-1), or all (0),'From':process (1...),or master (-1),'Command':'','Args':{'Arg1': 'Val1','Arg2': 'Val2'}}
            my_commandset = myctl.in_queue.get_nowait()
        except queue.Empty:
            continue
        try:
            # do not process if not To this process, or To: All  or from this process
            if (my_commandset['To'] != FLAGS.procnum and my_commandset['To'] != 0 and my_commandset['To'] != "master") or \
                    my_commandset['From'] == FLAGS.procnum:
                printandfile(
                    f"{myctl.procnum}-Commandset NG @ line {sys._getframe().f_lineno}, TO:{my_commandset['To']} From: {my_commandset['From']}  Commandset: {my_commandset}") # pylint: disable=W0212
                continue
        except KeyError:
            printandfile(f"{myctl.procnum}-Unaddressed command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
            continue
        retval, myctl, FLAGS = vidproc_process_queue(my_commandset, myctl, FLAGS)
        if not retval:  # Process_queue said bad coommandblock
            printandfile(f'{myctl.procnum}-Processqueue @ line {sys._getframe().f_lineno}, bad retval for {my_commandset}') # pylint: disable=W0212
            continue
    return (myctl, FLAGS)


def silence(ison):
    """
    Attempt to silence way too chatty tensorflow
    """
    if ison:

        #print("Silence is on")
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['FFREPORT'] = 'level=0:file=/var/log/ffmpeg.log'
        try:
            tf.logging.set_verbosity(logging.ERROR)
            tf.autograph.set_verbosity(0)
            tf.logging.set_verbosity(0)
        except NameError:
            pass

    else:
        #print("Silence is off")
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['FFREPORT'] = 'level=3:file=/var/log/ffmpeg.log'
        try:
            tf.logging.set_verbosity(3)
            tf.autograph.set_verbosity(3)
            tf.logging.set_verbosity(3)
        except NameError:
            pass



def getcap(cap, myctl, default):  # pylint: disable=W0613
    """
    Get value of capability cap in stream vid if available. Set to -1 if not
    """
    myval = -1
    global supplist
    if cap in supplist:
        myval = eval(f'myctl.vid.get(myctl.cv2.{cap})')
        return (myval)
    return (default)


def open_VideoWriter(myctl):
    """
    Make sure myctl.videofile exists, if not, create it and any parent dirs, and open it
    """
    checkopen(myctl.videofile, "w")
    return(myctl.cv2.VideoWriter(myctl.videofile, int(myctl.video_FourCC), myctl.video_fps, myctl.video_size))


def fileops(myctl, FLAGS, my_frame,out_pred_ext):
    """
    Video_process file operations
    """
    if myctl.is_recording:
        if myctl.out is None:  #has not been initialized
            myctl.framecounter = 0
            myctl.out = open_VideoWriter(myctl)
        if not myctl.out.isOpened():
            myctl.do_only_every = myctl.do_only_every_autovideo
            myctl.framecounter = 0
            myctl.out = open_VideoWriter(myctl )
        if FLAGS.record_framecode and (myctl.fcf is None or myctl.fcf.closed):
            myctl.fcf = checkopen(myctl.fcfile, "w", auto_open = True, buffering = 32768) #open framecode file with generous buffer
        # if FLAGS.record_framecode and myctl.fcf.closed:
        #    myctl.fcf = checkopen(myctl.fcfile, "w", auto_open = True, buffering = 32768) #open framecode file with generous buffer
        # write to the video file

        myctl.framecounter += 1
        myctl.out.write(my_frame)
        my_frame = None
        if FLAGS.record_framecode and out_pred_ext != []: #record framecode, but only if objects have been found
            myctl.fcf.write(f'{out_pred_ext},{myctl.framecounter}'+"\n")
    else:  # we aren't recording
        if myctl.out is not None:  # no action needed if not initialized
            if myctl.out.isOpened():
                myctl.do_only_every = myctl.do_only_every_initial
                myctl.framecounter = 0
                myctl.out.release()
                if myctl.presence_counter == -1:
                    try:
                        os.remove(myctl.videofile)
                    except FileNotFoundError:
                        pass
                    myctl.presence_counter = 0


        if FLAGS.record_framecode and myctl.fcf is not None:   # no action needed if not initialized
            if not myctl.fcf.closed:
                myctl.fcf.flush()
                myctl.fcf.close()
            try:
                if os.stat(myctl.fcfile).st_size == 0: # don't leave any empty fcfiles
                    os.remove(myctl.fcfile)
            except FileNotFoundError:
                pass
    return(myctl)

def rollavg(arrlen, value):
    """simple rolling average"""
    global workarr
    if len(workarr) >= arrlen:
        workarr.pop(0) #remove oldest
    workarr.append(value)
    if len(workarr) < arrlen:
        return(-1)
    return(sum(workarr)/len(workarr))

def video_file_type(myvideo_path):
    """
    Determine type of video file, and return
    """
    if isinstance(myvideo_path,int):
        return("webcam")
    if '://' in myvideo_path:
        return('stream')
    if myvideo_path.startswith('/') or  myvideo_path.startswith('\\'):
        return('file')
    return('file') #catchall

def handle_frame_fail(myctl):
    """
    What to do when video source fails after working before
    """
    #myctl = put_in_queue(myctl, From = myctl.procnum, To=-1, Command='status', Args={'StatusMsg': ["Video source stopped",'Video ended','Not ready']})
    myctl = vidproc_shutdown(myctl, what="Process shut down due to error", where = "Main loop",  why= 'No video')

def make_osd_suffix(myFLAGS,myctl):
    """
    Build a string suffix for OSD, based on record, autorecord, hush, ding, do_stats
    """
    suff = ""
    if myctl.record_autovideo:
        suff = f'{suff}Auto '
    if myctl.is_recording:
        suff = f'{suff}*REC* '
    if myFLAGS.hush:
        suff = f'{suff}H'
    else:
        suff = f'{suff}h'
    if myFLAGS.soundalert:
        suff = f'{suff}D'
    else:
        suff = f'{suff}d'
    if myctl.do_stats:
        suff = f'{suff}S'
    else:
        suff = f'{suff}s'
    return(suff)



def video_process(my_in_queue,my_out_queue,FLAGS, myevent, procnum):
    """
    Main video process. Captures video from webcam, file, or on-line stream. Runs video frames through specified YOLO model, displays and optionally stores the results.
    This process runs on its own GPU as specified in FLAGS.run_on_gpu, and/or it will run on a fraction of a GPU as specified in gpu_memory_fraction. There is no bounds checking,
    process will crash if GPU, or the total of gpu_memory_fraction per GPU are out of bounds. Each process can require around 2.5 G of resident memory, and it may crash
    ignominiously when memory-starved.
    Process will run forever unless stopped by operator, or a crashed video source.
    """
    global supplist
    printandfile(f'Video process {str(procnum).rjust(2)} start')
    if FLAGS.sync_start:
        myevent.clear()
    silence(FLAGS.hush)
    FLAGS.procnum = procnum  # record the passed process number
    ctl = makectl(FLAGS)  # set up a subset of FLAGS as multipurpose-object
    setattr(ctl, 'ding_timer', None)
    setattr(ctl, 'presence_timer', None)
    setattr(ctl, 'in_queue', my_in_queue)
    setattr(ctl, 'out_queue', my_out_queue)
    setattr(ctl, 'do_YOLO',True)
    atexit.register(vidproc_shutdown, ctl, echo = False)  # orderly shutdown, do not echo
    flagsok, reason,disposition = testparam(FLAGS) #check the flags sent to YOLO

    if not flagsok:
        ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': reason ,'Action': ""})
        if disposition == "stop":
            vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Init",  why= "Parameter err")
        if disposition == "no_YOLO":
            setattr(ctl, 'do_YOLO',False)


    if ctl.do_YOLO:
        ctl = put_in_queue(ctl, From = procnum, To=-1, Command='status', Args={'StatusMsg': ["Initializing YOLO", '','']})
        if FLAGS.monitor_YOLO:
            #print("Starting yt Timer")
            yt = threading.Timer(FLAGS.YOLO_init_AWOL_wait, yolo_timeout, [{'myctl' :ctl, 'what': "Process shut down due to error", 'where' : "During init", 'why': "YOLO init failed"}]) # set a timer for YOLO to initialize
            yt.start()
        #print(f'Video process {str(procnum).rjust(2)} init yolo <<<<<<<<<<<<<<<<<<<')
        setattr(ctl, 'yolo', init_yolo(FLAGS))
        #print(f'Video process {str(procnum).rjust(2)} done init yolo <<<<<<<<<<<<<<<<<<<')
        if FLAGS.monitor_YOLO:
            #print("Killing yt Timer")
            yt.cancel()
        ctl = put_in_queue(ctl, From = procnum, To=-1, Command='status', Args={'StatusMsg': ["Initializing YOLO ... success", '','']})
    else:
        ctl = put_in_queue(ctl, From = procnum, To=-1, Command='status', Args={'StatusMsg': ["Skipping YOLO", '','']})
        setattr(ctl, 'yolo', None)


    ################# for debug ####################################
    gc.enable()
    rtm = RepeatedTimer(300, writeobj,ctl)
    nfile = argv[0]
    nfile = nfile[:-3]+f"_objd_{ctl.procnum}" + ".log"
    with open(nfile, 'a+') as f:
        f.write("\n" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') +" STARTED" +"\n")
    writeobj(ctl) # do first one
    ################# for debug ####################################




    # check directories
    if not os.path.isdir(FLAGS.output_path):
        try:
            os.makedirs(FLAGS.output_path)
        except OSError:
            printandfile(f'{ctl.procnum}-Error @ line {sys._getframe().f_lineno}: Cannot create directory {FLAGS.output_path}') # pylint: disable=W0212
            vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Init",  why= "Directory err")
    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['',"Staging video",'']})


    if ctl.testing: # pylint: disable=E1101
        FLAGS.video_path = FLAGS.testvideo


    setattr(ctl, 'vid', cv2.VideoCapture(FLAGS.video_path)) # pylint: disable=E1101
    setattr(ctl, 'cv2', cv2)

    if not ctl.vid.isOpened() and not isinstance(FLAGS.video_path, int):  #can't open stream or vide file # pylint: disable=E1101
        ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot open stream or video file.",'Action': "Abandoning process"})
        time.sleep(1)
        vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Init",  why = 'No video')
    if not ctl.vid.isOpened() and isinstance(FLAGS.video_path, int):  #can't open a webcam # pylint: disable=E1101
        if FLAGS.video_path not in FLAGS.workingcam:
            ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot access cam {FLAGS.video_path}.",'Action': f"Try: {FLAGS.workingcam}"})
            time.sleep(1)
            vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Init", why = 'No cam')
        else:
            ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot access cam {FLAGS.video_path}.",'Action': ""})
            time.sleep(1)
            vidproc_shutdown(ctl, echo = False, why = 'No cam')

    if not ctl.vid.isOpened(): # pylint: disable=E1101
        # catchall
        printandfile(f'{ctl.procnum}-Error @ line {sys._getframe().f_lineno}: Cannot open video source {FLAGS.video_path}') # pylint: disable=W0212
        ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': f"Cannot open video source.",'Action': "Aborting"})
        time.sleep(1)
        vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Init", why = 'No video')

    if FLAGS.hush:
        with suppress_stdout_stderr():
            supplist = list_supported_capture_properties(ctl.vid, ctl.cv2)  # retrieve capabilities supported by stream/cam # pylint: disable=E1101
    else:
        supplist = list_supported_capture_properties(ctl.vid, ctl.cv2)  # retrieve capabilities supported by stream/cam  # pylint: disable=E1101

    setattr(ctl, 'Frame_Height', getcap('CAP_PROP_FRAME_HEIGHT', ctl, -1)) #getcap retrieves capability if supported. Default if not
    setattr(ctl, 'Frame_Width', getcap('CAP_PROP_FRAME_WIDTH', ctl, -1))

    # adjustment values for screens other than 1280x720, on which the current OSD layout is based. App will dynamically adjust to other sizes
    if ctl.Frame_Width > -1: # pylint: disable=E1101
        setattr(ctl, 'Type_Adjust', ctl.Frame_Width / 1280)  # create adjustment factor for screens other than 1280 wide # pylint: disable=E1101
        setattr(ctl, 'Line_Adjust', ctl.Frame_Height / 720)  # create adjustment factor for screens other than 720 high # pylint: disable=E1101
        setattr(ctl, 'Type_Thickness', int(round(3 * ctl.Type_Adjust, 0)))  # also adjust thickness if needed # pylint: disable=E1101
    else:
        setattr(ctl, 'Type_Adjust', 1)  # Leave as is if no width exposed
        setattr(ctl, 'Line_Adjust', 1)  # Leave as is if no width exposed
        setattr(ctl, 'Type_Thickness', 2)  # Leave as is if no width exposed

    setattr(ctl, 'video_FourCC', ctl.cv2.VideoWriter_fourcc("m","p","4","v"))  # pylint: disable=E1101


    if isinstance(FLAGS.video_path, int) and FLAGS.buffer_depth > 10 : #assume webcam, can't have buffer > 10
        FLAGS.buffer_depth = 10

    # long list of capability tests made necessary to support webcam that won't support most of these capabilities .....
    if 'CAP_PROP_BUFFERSIZE' in supplist:
        try:
            ctl.vid.set(ctl.cv2.CAP_PROP_BUFFERSIZE, FLAGS.buffer_depth)  # set very small buffer to combat video drift # pylint: disable=E1101
        except:
            pass

    myfps, myfpsmode = getoflfps(ctl, 0, 0) # pylint: disable=E1101
    setattr(ctl, 'init_fps', myfps)
    setattr(ctl, 'fps_mode', myfpsmode)

    #if ctl.fps_mode == 0: # pylint: disable=E1101
    #    put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['','','Preroll']})
    #    preroll(ctl, 24) # pylint: disable=E1101

    setattr(ctl, 'video_size', (
        int(ctl.vid.get(ctl.cv2.CAP_PROP_FRAME_WIDTH)), # pylint: disable=E1101
        int(ctl.vid.get(ctl.cv2.CAP_PROP_FRAME_HEIGHT)),)) # pylint: disable=E1101

    osdstr = "FPS: ??"
    prev_time = timer()
    curtype = ""
    setattr(ctl, 'init_frames', getcap('CAP_PROP_POS_FRAMES', ctl, -1))
    setattr(ctl, 'init_msec', getcap('CAP_PROP_POS_MSEC', ctl, -1))
    setattr(ctl, 'do_only_every', FLAGS.do_only_every_initial)

    ctl.ding_timer = RetriggerableTimer(0, ding_timeout)  # initialize, but don't start
    ctl.presence_timer = RetriggerableTimer(0, presence_timeout) # initialize, but don't start
    my_start_time = time.time()
    ctl.framecounter = itercounter = accum_time = curr_fps = sec_counter = 0
    x = 1

    ctl.out = ctl.fcf = None
    # Result log file
    if FLAGS.maintain_result_log:  # open the log file
        rslfile=f'{FLAGS.result_log_basedir}/{datetime.datetime.now().strftime("%y%m%d")}/P{ctl.procnum}/{datetime.datetime.now().strftime("%y%m%d_%H%M%S").strip()}_P{ctl.procnum}.csv'
        setattr(ctl, 'rsf',checkopen(rslfile, "w", auto_open = True, buffering = 32768)) #open running log file with generous buffer
        ctl.rsf.write("process,time,ytime,thrutime,rolltime,rollfps,objects,n,gpu,gpufract,allowgrowth,infps,outfps,waitkeytime,adj,out_pred\n\n") # pylint: disable=E1101

    if FLAGS.sync_start:  # stage video and wait for common start signal
        ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='videoready', Args={})  # tell master we are ready
        ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['','','Sync wait']})
        event_set = myevent.wait(FLAGS.sync_start_wait)  # now wait for event
        if event_set:
            pass
        else:
            printandfile(f"{ctl.procnum}-Time out, moving ahead without sync start ...")

    retval, frame = ctl.vid.read() # pylint: disable=E1101
    # Assume start at this point. Build offsets for CAP_PROP_POS_FRAMES and POS_MSEC, relative to this point
    msec_offset = getcap('CAP_PROP_POS_MSEC', ctl, -1)
    pos_offset  = getcap('CAP_PROP_POS_FRAMES', ctl, -1)

    if msec_offset == -1:
        init_msec = -1
    else:
        init_msec = 0

    if pos_offset == -1:
        init_pos = -1
    else:
        init_pos = 0

    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='initstats', Args={'InitPOS_FRAMES': init_pos,'InitPOS_MSEC': init_msec,'InitStrFPS': ctl.init_fps,'InitTimeStamp': datetime.datetime.timestamp(datetime.datetime.now()), 'run_on_gpu': ctl.run_on_gpu,'gpu_memory_fraction': ctl.gpu_memory_fraction}) # pylint: disable=E1101
    if FLAGS.profile:  # run profiler
        printandfile(f"{ctl.procnum} - Profiler activated.")
        pr = cProfile.Profile()
        pr.enable()

    global workarr #used for rolling average
    workarr =[]
    infps = roll = rollfps = 0

    setattr(ctl, 'file_type', video_file_type(FLAGS.video_path)) # stream, file, or USB?
    setattr(ctl, 'max_frames', int(getcap('CAP_PROP_FRAME_COUNT', ctl, -1)) ) # stream, file, or USB?

    if ctl.file_type == 'file'  and ctl.init_fps > 0: # pylint: disable=E1101
        waitkeytime = int((1/ctl.init_fps)*1000)  # pylint: disable=E1101
        do_adjust = True
    else:
        waitkeytime = 1
        do_adjust = False

    frame_fail_count = 0
    #send initial alive message

    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='imalive', Args = {})

    #==================================================================
    # Main video loop
    #==================================================================

    try:
        while ctl.vid.isOpened():  # pylint: disable=E1101
            tythru = 0 #execution timer
            mystr_1 = mystr_2 = ""
            frame = virgin_frame = None
            ctl,FLAGS = vidproc_checkqueue(ctl, FLAGS)  # check the queue, and act on it if necessary

            #grab a video frame
            try:
                retval, frame = ctl.vid.read()
            except KeyboardInterrupt as ki:
                printandfile(f"Video process {ctl.procnum} caught Ctl-c - shutting down")
                vidproc_shutdown(ctl, echo = False, what ="Process shut down intentionally",  where = "Main loop", why = 'Killed by user') # Shutdown and signal to all
            if not retval:  # error
                frame_fail_count += 1
                if frame_fail_count < ctl.init_fps *3: # tolerate 3 x framerate failures
                    ctl,FLAGS = vidproc_checkqueue(ctl, FLAGS)  # keep checking the queue
                    continue
                else:
                    #if we are playing a file, and if do_loop is set, play the file again
                    if ctl.file_type == 'file' and FLAGS.do_loop:
                        ctl.vid.set(cv2.CAP_PROP_POS_FRAMES, 1)
                    else:
                        handle_frame_fail(ctl)
            virgin_frame = frame # save unadorned frame
            frame_fail_count = 0  #reset frame_fail_count
            try:
                frame = frame[:, :, :: 1]
            except TypeError:
                continue


            #logic to run only every nth frame through yolo to combat frame drift, and to save energy
            if (itercounter % ctl.do_only_every == 0 or ctl.do_only_every == -1) and ctl.do_YOLO:  # do only every n frame, -1 is each. Skip if no YOLO
                curtype = curlabel = ""
                image = Image.fromarray(frame)
                if FLAGS.monitor_YOLO:
                    yt2 = threading.Timer(FLAGS.YOLO_detect_AWOL_wait, yolo_timeout, [{'myctl' :ctl, 'what': "Process shut down due to error", 'where' : "Main loop", 'why': "YOLO detect fail"}]) # set a timer for YOLO to initialize
                    yt2.start()

                typre = timer() #time actual execution
                try:
                    image, elapsed_time,  out_pred_ext = ctl.yolo.detect_image_extended(image, show_stats=FLAGS.show_stats)
                    tynow = timer()
                    tythru = tynow - typre
                    #print(f"{typre}  {tynow}  {tythru}")
                except RuntimeError:
                    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='error', Args={'ErrMsg': 'YOLO runtime error' ,'Action': "Shutting down"})
                    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['YOLO runtime error', '','Shutdown']})
                    vidproc_shutdown(ctl, echo = False, what="Process shut down due to error", where = "Main loop",  why= "YOLO runtime err")

                #if FLAGS.monitor_YOLO and not (itercounter == ctl.video_fps * 1 and ctl.procnum == 3) :
                if FLAGS.monitor_YOLO:
                    yt2.cancel()

                #build rolling average to determine maxmimum frame rate this YOLO instance is capable of
                if ctl.do_YOLO:
                    roll=rollavg(FLAGS.rolling_average_n,elapsed_time)
                else:
                    roll = 0
                if roll > 0:
                    rollfps = round(1/roll,1)
                else:
                    rollfps = 0
                result = np.asarray(image)
            else:
                result = frame
                out_pred_ext = []
                elapsed_time = 0

            #logic to determine fps
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps += 1
            cur_posframes = getcap('CAP_PROP_POS_FRAMES', ctl, -1)
            cur_posmsec = getcap('CAP_PROP_POS_MSEC', ctl, -1)


            if accum_time > 1:  #happens each second

                ctl.video_fps = curr_fps  # set output stream to the real fps this is running at
                accum_time -= 1


                #Build OSD line
                if rollfps > 0:
                    myyfps = str(rollfps)
                else:
                    myyfps = CONST_DASHDASHDASH

                osdstr = f"IFPS: {str(round(infps,1))} YFPS: {myyfps} OFPS: {str(round(ctl.video_fps,1))} RT: {str(round(roll,3))} n={str(ctl.do_only_every)} {make_osd_suffix(FLAGS,ctl)}"
                curr_fps = 0
                # send to master
                if cur_posframes != -1:
                    posframes = cur_posframes - pos_offset
                if cur_posmsec != -1:
                    posmsec = cur_posmsec - msec_offset


                if ctl.fps_mode == 0: #if PROP_FPS works, take it continuously
                    cur_infps = getcap('CAP_PROP_FPS', ctl, -1)
                if ctl.fps_mode == 1: #rely on POS_FRAMES/POS_MSEC, provides average only
                    cur_infps =  round((ctl.vid.get(cv2.CAP_PROP_POS_FRAMES) - ctl.init_frames) / (ctl.vid.get(cv2.CAP_PROP_POS_MSEC) - ctl.init_msec) * 1000, 1) # pylint: disable=E1101
                if ctl.fps_mode == 2: #Can't do timing loop each time, so rely on what was measured at start of stream
                    cur_infps = ctl.init_fps


                #report stats to master
                if ctl.do_stats:
                    ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='stats',
                             Args={'POS_FRAMES': getcap('CAP_PROP_POS_FRAMES', ctl, -1) -pos_offset,
                                   'POS_MSEC': getcap('CAP_PROP_POS_MSEC', ctl, -1) - msec_offset,
                                   'MaxYoloFPS': rollfps,
                                   'InFPS': cur_infps,
                                   'StrFPS': ctl.video_fps,
                                   'DoOnlyEvery' : ctl.do_only_every,
                                   'TimeStamp': datetime.datetime.timestamp(datetime.datetime.now()),
                                   'Looping': FLAGS.do_loop}
                                   )
                #send alive msg
                ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='imalive', Args = {})

                sec_counter += 1
                if FLAGS.maintain_result_log and  sec_counter > 60 :
                    ctl.rsf.flush() # flush every 60 secs
                    sec_counter = 0

            # update incoming FPS. Mode was set when setting up stream
            if ctl.fps_mode == 0: #if PROP_FPS works, take it continuously
                infps = getcap('CAP_PROP_FPS', ctl, -1)
            if ctl.fps_mode == 1: #rely on POS_FRAMES/POS_MSEC, provides average only
                infps = round((ctl.vid.get(cv2.CAP_PROP_POS_FRAMES) - ctl.init_frames) / (ctl.vid.get(cv2.CAP_PROP_POS_MSEC) - ctl.init_msec) * 1000, 1) # pylint: disable=E1101
            if ctl.fps_mode == 2: #Can't do timing loop each time, so rely on what was measured at start of stream
                infps = ctl.init_fps
            outfps = ctl.video_fps

            # governor - adjust waitkey time to match incoming fps
            if do_adjust:
                diff = abs(int(outfps)-int(infps))
                adj = 1
                if diff > 5:
                    adj = 20
                if diff > 3:
                    adj = 10
                if int(outfps) < int(infps) and waitkeytime > adj:
                    adj = adj * -1
                if int(outfps) == int(infps):
                    adj = 0
                if waitkeytime + adj > 0:
                    waitkeytime = waitkeytime + adj
            else:
                adj = 0

            #result log
            if FLAGS.maintain_result_log and roll > 0: #write to result log, but only if we have a valid rolling average
                statsline= f'{ctl.procnum},\'{datetime.datetime.now().strftime("%H-%M-%S_%f")}\',{round(elapsed_time,5)},{round(tythru,5)},{round(roll,5)},{round(1/roll,1)},{len(out_pred_ext)},{ctl.do_only_every},{FLAGS.run_on_gpu},{FLAGS.gpu_memory_fraction},{FLAGS.allow_growth},{infps},{ctl.video_fps}, {waitkeytime},{adj},{out_pred_ext}\n'
                ctl.rsf.write(statsline)

            #presence trigger, autovideo
            if out_pred_ext != []:

                #BS Special __ find out if y < 80 ##############
                trigger_OK = True
                for pred in out_pred_ext:
                    if pred[1] < 72:
                        trigger_OK = False
                ################################################


                labstring = curtype = curlabel = ""

                lablist = []
                # extract labels
                for pred in out_pred_ext:
                    lablist.append(pred[4])

                # now we have all possible labels in lablist
                for curlabel in lablist:
                    curtype = FLAGS.labeldict.get(curlabel,'notype')

                    if curtype in FLAGS.presence_trigger:  # check the access list

                        if FLAGS.soundalert:
                            if not ctl.ding_timer.is_alive():
                                playsound(FLAGS)
                                ctl.ding_timer = RetriggerableTimer(FLAGS.ding_interval, ding_timeout)  # set-up the timer
                                ctl.ding_timer.start()
                            else:
                                ctl.ding_timer.reset()
                        ctl.presence_counter += 1
                        if not ctl.presence_timer.is_alive():
                            if ctl.record_autovideo and trigger_OK:
                                ctl = start_recording(ctl, putqueue=True) #if putqueue true, cause all processes to record
                                ctl.autovideo_running = True
                            ctl.presence_timer = RetriggerableTimer(FLAGS.presence_timer_interval,
                                                                presence_timeout)  # set-up the timer
                            ctl.presence_timer.start()
                        else:
                            ctl.presence_timer.reset(FLAGS.presence_timer_interval)

            osdkwarg = {"fontFace": 0, "fontScale": 1.2 * ctl.Type_Adjust, "color": (0, 255, 0),
                        "thickness": int(ctl.Type_Thickness)}

            #file it away
            if FLAGS.record_with_boxes:
                ctl = fileops(ctl,FLAGS, virgin_frame, out_pred_ext) #save frame to file, without OSD, without boxes
            else:
                ctl = fileops(ctl,FLAGS, result,out_pred_ext) #save frame to file, without OSD, with boxes


            #show OSD
            if ctl.osd and ctl.do_YOLO: #only show OSD if enabled, and we are doing YOLO
                result = ctl.cv2.putText(result, text=osdstr, org=(20, int(ctl.video_size[1] - 20 * ctl.Line_Adjust)   ),**osdkwarg   )

            if not ctl.do_YOLO:
                result = ctl.cv2.putText(result, text="No YOLO model specified," , org=(20, int(ctl.video_size[1] - 105 * ctl.Line_Adjust)),**osdkwarg)
                result = ctl.cv2.putText(result, text="or YOLO model not found.", org=(20, int(ctl.video_size[1] - 65 * ctl.Line_Adjust)),**osdkwarg)
                result = ctl.cv2.putText(result, text="TrainYourOwnYOLO!", org=(20, int(ctl.video_size[1] - 20 * ctl.Line_Adjust)   ),**osdkwarg   )


            #show video

            ctl.cv2.namedWindow(FLAGS.window_title, ctl.cv2.WINDOW_NORMAL)
            ctl.cv2.moveWindow(FLAGS.window_title, FLAGS.window_x, FLAGS.window_y)
            ctl.cv2.resizeWindow(FLAGS.window_title, FLAGS.window_wide, FLAGS.window_high)
            ctl.cv2.imshow(FLAGS.window_title, result) #show the frame

            _ = ctl.cv2.waitKey(waitkeytime)

            try:
                # check for window killed
                if ctl.cv2.getWindowProperty(FLAGS.window_title, ctl.cv2.WND_PROP_VISIBLE) <1:
                    printandfile(f"Process {ctl.procnum} was killed")
                    vidproc_shutdown(ctl, echo = False, what ="Process shut down intentionally",  where = "Main loop", why = 'Killed by user')
            except cv2.error:
                pass


            # check for autovideo no longer running
            if not ctl.presence_timer.is_alive() and ctl.autovideo_running:  #we need to switch it off
                ctl = stop_recording(ctl, putqueue=True, delete= ctl.presence_counter < 5)
                ctl.presence_counter = 0

            itercounter += 1
            if (time.time() - my_start_time) > x:
                my_start_time = time.time()

            if itercounter == ctl.video_fps * 1: # Signal video up and running after 5 seconds of good video
                #put_in_queue(ctl, From = ctl.procnum, To=-1, Command='status', Args={'StatusMsg': ['Process active','          ','         ']})
                ctl = put_in_queue(ctl, From = ctl.procnum, To=-1, Command='activate', Args={}) #Signal that this process has an active video stream


            #if ctl.procnum == 3:
            #    while True:
            #        pass

            if itercounter > 1000000:
                itercounter = 0

            if itercounter == 3000 and FLAGS.profile:
                break

        if FLAGS.profile:
            # actions if profiler enabled
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            proffile = getbasefile()+f"_{ctl.procnum}.prof"
            ps.print_stats()
            logout=f'Proc {ctl.procnum} {s.getvalue()}'
            with open(proffile, "a+") as f:
                f.write(logout)
            print(logout)
            time.sleep(20) #wait to give all processes a chance to save
            print(f"{ctl.procnum} - Profiler deactivated.")

    except KeyboardInterrupt as ki:
            printandfile(f"Video process {ctl.procnum} caught Ctl-c - shutting down")
            vidproc_shutdown(ctl, echo = False, what ="Process shut down intentionally",  where = "Main loop", why = 'Killed by user') # Shutdown and signal to all

    # should never get here, just in case
    vidproc_shutdown(ctl, echo = True)  # shutdown, and echo to others
    try:
        ctl.rsf.close()
    except:
        pass


def put_in_queue(myctl,From=0, To=0, Command='', Args=None):
    """
    Put command and args in queue, send to To
    """
    if Args is None:
        Args = {}
    my_commandset = {'To': To, 'From': From , 'Command': Command, 'Args': Args}

    # if not sent from master, send to master for distribution
    if myctl.procnum != -1:
        myctl.out_queue.put(my_commandset)
    else:
        # if sent from master, distribute immediately
        myctl = process_master_commandset(myctl, my_commandset)
    return(myctl)

def vidproc_process_queue(my_commandset, myctl, FLAGS):
    """
    This processes a command block coming from master into video process
    returns retval True/False (False = error) and myctl
    """
    if not isinstance(my_commandset, dict):
        printandfile(f"{myctl.procnum}-Queue sent bad command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
        return (False, myctl, FLAGS)
    try:
        mycommand = my_commandset['Command'].strip()
        myarglist = my_commandset['Args']
    except KeyError:
        printandfile(f"{myctl.procnum}-Queue sent bad command block @ line {sys._getframe().f_lineno}, ignoring") # pylint: disable=W0212
        return (False, myctl, FLAGS)
    # work the commands, set putqueue=False to prevent cascading queues
    if mycommand == "shutdown":
        myctl = stop_recording(myctl, putqueue=False, delete=False)
        myctl = put_in_queue(myctl,  From = myctl.procnum, To=-1, Command='ackshutdown', Args={})
        time.sleep(1)
        vidproc_shutdown(myctl, echo = False, what=f"Received shutdown command from proc {my_commandset.get('From',-1)}", where = "Main loop", )
        sys.exit()
    if mycommand == "startrecording":
        myctl = start_recording(myctl, putqueue=False)
        return (True, myctl, FLAGS)
    if mycommand == "rualive":
        myctl = put_in_queue(ctl, From = ctl.procnum, To=my_commandset.get('From',-1), Command='imalive', Args = {})
        return (True, myctl, FLAGS)
    if mycommand == "stoprecording":
        myctl = stop_recording(myctl, putqueue=False, delete=myarglist.get("Delete",False))
        return (True, myctl, FLAGS)

    if mycommand == "autovidoff":
        myctl = stop_recording(myctl, putqueue=False, delete=False)
        myctl.record_autovideo = False
        FLAGS.record_autovideo = False
        return (True, myctl, FLAGS)

    if mycommand == "autovidon":
        myctl.record_autovideo = True
        FLAGS.record_autovideo = True
        return (True, myctl, FLAGS)

    if mycommand == "abortvideowriter":
        myctl.is_recording = False
        if myctl.out is not None and myctl.out.isOpened():
            myctl.out.release()
        return (True, myctl, FLAGS)
    if mycommand == "audioon":
        FLAGS.soundalert = True
        playsound(FLAGS)
        return (True, myctl, FLAGS)
    if mycommand == "audiooff":
        FLAGS.soundalert = False
        return (True, myctl, FLAGS)
    if mycommand == "hushoff":
        FLAGS.hush = False
        return (True, myctl, FLAGS)
    if mycommand == "hushon":
        FLAGS.hush = True
        return (True, myctl, FLAGS)
    if mycommand == "osdoff":
        myctl.osd = False
        return (True, myctl, FLAGS)
    if mycommand == "osdon":
        myctl.osd = True
        return (True, myctl, FLAGS)
    if mycommand == "statson":
        myctl.do_stats = True
        writethelog("Stats ON",myctl.procnum)
        return (True, myctl, FLAGS)
    if mycommand == "statsoff":
        myctl.do_stats = False
        writethelog("Stats OFF",myctl.procnum)
        return (True, myctl, FLAGS)

    if mycommand == "queuetest":
        printandfile(f"{myctl.procnum}-Would execute queuetest")
    if mycommand == "windowredraw":
        try:
            myctl.cv2.moveWindow(FLAGS.window_title, FLAGS.window_x, FLAGS.window_y - 50)
        except cv2.error:
            pass
    return (True, myctl, FLAGS)



def update_lastseendict(themctl,theproc):
    """
    Update themctl.lastseendict with last seen proc and time of sightinmg
    """
    themctl.lastseendict[theproc] = datetime.datetime.utcnow()
    return(themctl)

def check_lastseendict(themctl, seconds_late):
    """
    Scan lastseendict for processes that haven't been seen in the past seconds_late seconds
    Return True if all accounted for, False and a list of possible dead processes
    """
    deadlist=[]
    lastseen = None
    for key in themctl.procdict:
        if key == -1 or get_proc_prop(themctl,key, 'is_running') == False:
            continue
        #now we have a process that should be awake
        lastseen = themctl.lastseendict.get(key,None)
        if lastseen != None:
            #if process was never heard from, or heard from for more that 15 sec ago, report the process
            islate = int(round(datetime.timedelta.total_seconds(datetime.datetime.utcnow()-lastseen),0))
        else:
            islate = 0
        if lastseen == None or islate > seconds_late:
            deadlist.append([key,islate])
    if deadlist == []:
        return(True, deadlist)
    else:
        return(False, deadlist)

def check_for_hung_process(themctl):
    global showmaster, shutting_down  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    assert(themctl != None)
    """
    Check for unresposive process. If found, try to wake up. If no reaction, declare dead.
    """
    if len(themctl.lastseendict) < themctl.num_processes:
        #all processes no up yet
        assert(themctl != None)
        return(themctl)
    allok, deadbeats = check_lastseendict(themctl,5)
    if allok:
        assert(themctl != None)
        return(themctl)
    else:
        if themctl.all_running: #show only if process up & running
            #issue warning only
            for deadbeat in deadbeats:
                if get_proc_prop(themctl,deadbeat[0], 'is_running'):
                    msg=f"Process lagging by {deadbeat[1]} seconds"
                    do_dataline(themctl, deadbeat[0], emerg = msg)
                    #print(f"Statusline:  {[msg,'          ','          ']}")
                    do_statusline(themctl, deadbeat[0], [msg,'          ','          '])

    allok, deadbeats = check_lastseendict(themctl,15)
    if allok:
        assert(themctl != None)
        return(themctl)
    #work the deadbeats
    for deadbeat in deadbeats:
        #kill the process
        writethelog(f"Master: Process {deadbeat[0]} declared deadbeat, pronounced_dead, terminated")
        #tally(themctl)
        themctl = killproc(themctl, deadbeat[0])
        if showmaster:
            assert(themctl != None)
            do_dataline(themctl, deadbeat[0], emerg = f"Process hung for {deadbeat[1]} sec, terminated")
            assert(themctl != None)
            do_statusline(themctl, deadbeat[0], [f"Process hung for {deadbeat[1]} sec, terminated",'          ','          '])
        assert(themctl != None)
    return(themctl)

def killproc(themctl, theproc):
    writethelog(f"Master: Process {theproc} killed by killproc")
    get_proc_prop(themctl,theproc, 'process').terminate()  #kill the process
    themctl = put_proc_prop(themctl,theproc, 'is_running', False)
    themctl = put_proc_prop(themctl,theproc, 'is_active', False)
    themctl = put_proc_prop(themctl,theproc, 'is_pronounced_dead', True)
    return(themctl)


def tally(themctl):

    in_procdict = in_lastseendict = is_running = is_pronounced_dead = is_active = is_videoready = 0
    in_procdict = len(themctl.procdict) -1
    in_lastseendict = len(themctl.lastseendict)
    for key in themctl.procdict:
        if key == -1:
            continue
        if get_proc_prop(themctl,key, 'is_running'):
            is_running += 1
        if get_proc_prop(themctl,key, 'is_pronounced_dead'):
            is_pronounced_dead += 1
        if get_proc_prop(themctl,key, 'is_active'):
            is_active += 1
        if get_proc_prop(themctl,key, 'is_videoready'):
            is_videoready += 1

    print(f"in_procdict: {in_procdict} in_lastseendict: {in_lastseendict} is_running: {is_running}  is_pronounced_dead: {is_pronounced_dead}  is_active: {is_active}  is_videoready: {is_videoready}")
    print(f"in_procdict: {in_procdict} in_lastseendict: {in_lastseendict} is_running: {get_tot_proc_prop(themctl, 'is_running')}  is_pronounced_dead: {get_tot_proc_prop(themctl, 'is_pronounced_dead')}  is_active: {get_tot_proc_prop(themctl, 'is_active')}  is_videoready: {get_tot_proc_prop(themctl, 'is_videoready')} ")
    #print(f"Videoready: {get_tot_proc_prop(themctl, 'is_videoready')}")



def tally2(themctl):
    for key in themctl.procdict:
        if key == -1:
            continue





def process_master_commands(my_commandset, my_mctl):
    global do_update, showmaster
    """
    Act on commands sent to master by video_processes
    """
    procnumber = None
    mycommand = my_commandset.get('Command','').strip()
    myarglist = my_commandset.get('Args',{})
    myproc = my_commandset.get('From', None)
    iam = str(myproc)
    #print(f"Master command: {mycommand} from: {myproc} ")
    if mycommand == "procdied": # We've lost a process
        print(f"Master: Received procdied, proc {myproc} ")
        # Ack death
        writethelog(f"Master: Received procdied, reason {myarglist.get('what','')} {myarglist.get('where','')} {myarglist.get('why','')} from process {iam}. Killing {iam}")

        #tally(my_mctl)
        if 0 >= myproc > my_mctl.procnum: #Bogus
            printandfile(f"Error @ line {sys._getframe().f_lineno} in dead process report, reported procnum: {myproc}. Resuming, but taking no action") # pylint: disable=W0212
            return(my_mctl) # Do nothing
        my_mctl.initready_counter -= 1
        # immediately release any staged video processes
        my_mctl.event.set()
        my_mctl.sync_start = False
        # Announce the death of the process
        if showmaster:
            do_dataline(my_mctl, myproc, emerg = "Process defunct")
            do_statusline(my_mctl, myproc, [myarglist.get('what',''), myarglist.get('where',''),myarglist.get('why','')])
        my_mctl = killproc(my_mctl, myproc)
        return(my_mctl)

    if mycommand == 'activate': #process is active
        #set it active in procdict
        my_mctl = put_proc_prop(my_mctl,myproc, 'is_active', True)
        if showmaster:
            do_statusline(my_mctl, myproc, ['Process active','Main loop','All O.K.'])
        return(my_mctl)


    if mycommand == 'shuttingoff': #process declared it is shutting down
        my_mctl = killproc(my_mctl, myproc)
        # immediately release all staged video processes
        my_mctl.event.set()
        writethelog(f"Received shutting off from process {myproc} terminating")
        printandfile(f"Terminating process {myproc}")
        #tally(my_mctl)
        return(my_mctl)

    if mycommand == 'status':
        #my_mctl = update_lastseendict(my_mctl,myproc)
        if showmaster:
            mymsgr = myarglist.get('StatusMsg', [])
            do_statusline(my_mctl, myproc, mymsgr)
        return(my_mctl)

    if mycommand == 'imalive':
        my_mctl = update_lastseendict(my_mctl,myproc)
        return(my_mctl)


    if mycommand == 'error'  and showmaster:

        pad = ' ' * 70  #70 spaces
        mymsg = (f"Proc{str(iam).rjust(3)}: {myarglist.get('ErrMsg','')} {myarglist.get('Action','')}" +pad)[:70]
        writethelog(f"Master: Received error {mymsg} from process {iam}")
        mycol = 0
        myw = my_mctl.etab
        # message = (f"Proc#{str(iam).rjust(2)}: {mymsg} - {myarglist.get('Action','')}" + pad)[:40]
        settext(myw,mymsg,my_mctl.next_etab_statusline,1)
        my_mctl.next_etab_statusline = my_mctl.next_etab_statusline +1
        ### myw.update()
        return(my_mctl)



    if mycommand == 'initstats'  and showmaster:
        my_mctl.initdict[iam] = myarglist
        my_mctl.initready_counter -= 1
        if my_mctl.initready_counter == 0:  # we have all initial stats
            my_mctl = do_masterscreen(my_mctl)
        return (my_mctl)

    if mycommand == 'stats' and showmaster and my_mctl.masterup:
        my_mctl.statsdict[iam] = myarglist
        my_mctl = do_dataline(my_mctl, iam)  # from now on, write single lines
        if do_update:
            try:
                my_mctl.window.update()
            except: #tkinter does not return proper error type
                pass
            do_update = False
        return(my_mctl)


    if mycommand == 'pickle':
        with open("/usr/local/bin/catdetectorp/TrainYourOwnYOLO/3_Inference/mctl.pkl", "wb") as f:
            pickle.dump(my_mctl, f, protocol=pickle.HIGHEST_PROTOCOL)
        return (my_mctl)
    if mycommand == 'videoready':
        if not my_mctl.sync_start:
            #my_mctl.event.set()
            pass
        else:
            my_mctl = put_proc_prop(my_mctl, myproc, 'is_videoready', True)
            #tally(my_mctl)
            if is_video_ready(my_mctl):
                my_mctl.event.set()
                my_mctl.sync_start = False
        return (my_mctl)

    if mycommand == 'ackshutdown':  #Wait for all video_processes signaling shutdown, then shutdown master
        my_mctl =  killproc(my_mctl,myproc)
        do_statusline(my_mctl, myproc, ["Process shut down intentionally","Main loop",'Killed by master'])
        time.sleep(0.3)
        if are_all_dead(my_mctl):
            if my_mctl.shutdown_action == -1: # master shutdown
                writethelog(f"All processed did ack shutdown request, shutting down")
                master_shutdown(my_mctl, why = "Master shutdown")

            if my_mctl.shutdown_action == 1: # resurrect
                writethelog(f"All processed did ack shutdown request, restarting")
                master_shutdown(my_mctl, why = "Master restart", restart = True)
        return(my_mctl)




def is_video_ready(my_mctl):
    """return True if all proecesses, not counting dead, are video_ready"""
    allproc     = len(my_mctl.procdict) - 1
    allvidready = get_tot_proc_prop(my_mctl, 'is_videoready')
    alldead     = get_tot_proc_prop(my_mctl, 'is_pronounced_dead')
    if allvidready + alldead >= allproc:
        return True
    else:
        return False

def are_all_dead(my_mctl):
    """return True if all proecesses dead"""
    allproc     = len(my_mctl.procdict) - 1
    alldead     = get_tot_proc_prop(my_mctl, 'is_pronounced_dead')
    if allproc - alldead <= 0:
        return True
    else:
        return False





def converttype(myval,linenum):
    """
    Massage conf file settings
    """
    # check for string
    myval = myval.strip()  # cleanup
    if myval.startswith('"') and myval.endswith('"'):
        return (myval.strip('"'))  # assume string
    if myval.startswith("'") and myval.endswith("'"):
        return (myval.strip("'"))  # assume string
    if (myval.startswith("{") and myval.endswith("}")) or (myval.startswith("[") and myval.endswith("]")) or (
            myval.startswith("(") and myval.endswith(")")):
        return (ast.literal_eval(myval))  # assume dict, list, tuple
    if (myval.startswith("{") and not myval.endswith("}")) or (not myval.startswith("{") and myval.endswith("}")) or (
            myval.startswith("[") and not myval.endswith("]")) or (
            not myval.startswith("[") and myval.endswith("]")) or (
            myval.startswith("(") and not myval.endswith(")")) or (not myval.startswith("(") and myval.endswith(")")):
        printandfile(f"Error in config file @ line {linenum}, unbalaced paren in {myval}. Source line {sys._getframe().f_lineno}") # pylint: disable=W0212
        sys.exit()

    if myval.isdecimal():
        return (int(myval))  # assume integer

    if myval == 'True' or myval == 'False': # assume boolean
        return (ast.literal_eval(myval))

    #test for float
    try:
        return(float(myval))  # try float
    except:
        printandfile(f"Error in config file @ line {linenum}, bad parameter {myval}. Source line {sys._getframe().f_lineno}") # pylint: disable=W0212
        sys.exit()


def getbasefile():
    """
    returns path to current app with ".py" stripped off
    """
    basefile = __file__  # get path to app
    if basefile.endswith('.py'):
        basefile = basefile[:-3]
    return(basefile)

def makeconfig():
    """
    This provides the settings for the master module and the individual processes
    For this, we use a config file with the same app name and the ".conf" extension, situated in the app directory
    If there is no config file, the app will default to one process using the test video file and the common settings/directories of TrainYourOwnYolo
    """
    # convert config file into blk_Master, blk_Common, and blk_process_block_dict objects

    confpath = getbasefile() + '.conf'
    # assert os.path.exists(confpath), f'Cannot find conf file {confpath}'
    if os.path.exists(confpath):

        # if there is a config file, update defaults from config
        with open(confpath, 'r') as f:
            conf = f.read()
        confl = conf.splitlines()  # read the file into memory, turned into a list
        has_config = 1 #Config file good, so far
    else:
        has_config = -1 #Tell Master we are working without config file
        confl = []

    myblk_Master, myblk_Master_OK = getblock("Master:", confl,
                      DefMaster())  # add all entries in Master block of the confl list to an (empty) CTL object, and store as blk_Master

    myblk_Common, myblk_Common_OK = getblock("Common:", confl,
                      DefCommon())  # add all entries in Common block of the confl list to an (empty) CTL object, and store as blk_Common

    if has_config > -1:
        if not myblk_Master_OK  and not myblk_Common_OK:
            setattr(myblk_Master,'has_config', -2) #Config file empty
        else:
            setattr(myblk_Master,'has_config', -3) #Config file partial
        if myblk_Master_OK  and myblk_Common_OK:
            setattr(myblk_Master,'has_config', 1) #Good config file
    else:
        setattr(myblk_Master,'has_config', -1) #No config file

    if myblk_Master.hush: #try to silence chatty tensorflow
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['FFREPORT'] = 'level=0:file=/var/log/ffmpeg.log'
    else:
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['FFREPORT'] = 'level=32:file=/var/log/ffmpeg.log'
    silence(myblk_Master.hush)
    setattr(myblk_Common, "hush", myblk_Master.hush) # propagate hush to Common
    if myblk_Common.soundalert:  # test audio output if sound requested
        if not soundtest(myblk_Common):  # if output fails, disable sound
            print("No audio output found, disabling chime")
            myblk_Common.soundalert = False #This disables soundalert in all processes
            setattr(myblk_Master, "soundavailable", False)
        else:
            setattr(myblk_Master, "soundavailable", True)

    if hasattr(myblk_Master, 'sync_start'):
        if myblk_Master.num_processes == 1: # Ignore sync start if only 1 process
            setattr(myblk_Common, "sync_start", False)
            setattr(myblk_Master, "sync_start", False)
        else:
            setattr(myblk_Common, "sync_start", myblk_Master.sync_start)
            setattr(myblk_Common, "sync_start_wait", myblk_Master.sync_start_wait * myblk_Master.num_processes)  #wait sync_start_wait multiplied by num_processes for sync start
    else:
        setattr(myblk_Common, "sync_start", False)


    if myblk_Master.hush: #suppress messages when hushed
        with suppress_stdout_stderr():
            avlcam, workingcam = listcams(myblk_Common.max_cams_to_probe)
    else:
        avlcam, workingcam = listcams(myblk_Common.max_cams_to_probe)
    setattr(myblk_Common,'avlcam',avlcam) # get available usb/internal camera indices
    setattr(myblk_Common,'workingcam',workingcam) # get working usb/internal camera indices
    # grab a few Common settings to be used by Master
    setattr(myblk_Master, 'record_autovideo', myblk_Common.record_autovideo)
    setattr(myblk_Common, 'do_stats', myblk_Master.do_stats)
    setattr(myblk_Master, 'soundalert', myblk_Common.soundalert)
    if not hasattr(myblk_Common, 'testvideo') or myblk_Common.testvideo == "":
        setattr(myblk_Common,'testvideo', resource_path + "/testvid.mp4")
    if not hasattr(myblk_Common, 'dingsound') or myblk_Common.dingsound== "":
        setattr(myblk_Common,'dingsound', resource_path + "/chime.mp3")
    if not hasattr(myblk_Common, 'silence') or myblk_Common.silence == "":
        setattr(myblk_Common,'silence', resource_path + "/silence.mp3")
    myblk_process_block_dict = get_process_block_dict(confl, myblk_Common,
                               myblk_Master.num_processes)  # get dict of Process  blocks, each added to properties in blk_common dict is {"procnum": block object}

    return (myblk_Master, myblk_process_block_dict)

def find_between(mystart,myend,mystring):
    """Return string between mystart and myend. Empty if mystring  won't start with mystart or end with myend"""
    if not mystring.startswith(mystart):
        return('')
    if not mystring.endswith(myend):
        return('')
    return(mystring[mystring.find(mystart)+len(mystart):mystring.rfind(myend)])

def is_new_block(myli):
    """
    Return True for start of new block, False if not)
    """
    try:
        return((myli[0].startswith('Master') or myli[0].startswith('Common') or myli[0].startswith('Process_')) and myli[1] == '')
    except IndexError:
        return(False)

def getblock(blockname, confl, addobj):
    """
    Get config block specified in blockname from config list confl, add to blockobj, and return blockobj
    """
    try: #confl is the whole config file, split into a list
        i = confl.index(blockname) # get the position of the block we are looking for
    except ValueError:
        return(addobj,False) # return the default block, no success
    except NameError:
        printandfile(f"Config file failed @ line {sys._getframe().f_lineno}") # pylint: disable=W0212
        return(addobj,False) # return the default block, no success
    blockobj = copy.deepcopy(addobj)
    while True:
        i += 1
        if i >= len(confl):  # end of file!
            return (blockobj,True)


        #check for block marker without colon
        if (confl[i].strip().startswith('Master') or confl[i].strip().startswith('Common') or confl[i].strip().startswith('Process_')) and not confl[i].strip().endswith(':'): #Block marker needs a colon
                printandfile(f"Config file error at line number {i+1}: Possible block marker '{confl[i].strip()}' does not end in a ':' . Source line {sys._getframe().f_lineno}")
                sys.exit()
        if confl[i].strip().startswith('#') or ':' not in confl[i]:  # skip comment, or line without :, therefore also empty line
            continue
        sl = confl[i].strip().split('#', 1)[0].strip()  # chop off trailing comment
        li = sl.strip().split(':', 1)  # split at colon, one split
        if is_new_block(li):
            return (blockobj,True)
        theprop = li[0].strip()
        theval = li[1].strip()

        if theval == '': #empty val, don't take it
            printandfile(f"Empty config file setting '{theprop}' at line number {i+1}. Using default instead.<<<<<<<<<")
        else:
            cval = converttype(theval, i+1)  # convert the type
            setattr(blockobj, theprop, cval)  # .. and add to the blockobject, blockobject.theprop = cval
    return (blockobj,True)  # just to make sure, should never see this


def get_process_block_dict(confl, blk_Common, num_proc):
    """
    get dict of numproc process blocks, each added to properties in blk_common dict is {"procnum": block object}.
    get config block specified in blockname from config list confl, add to blockobj, and return blockobj
    """
    my_process_block_dict = {}

    procsfound = [i for i in confl if i.startswith('Process_')] # list all Process_ blocks

    for procblk in procsfound: #go through all Process_ blocks

        try:
            j = int(find_between('Process_',':',procblk))
        except ValueError: # procblk failed
            continue
        try:
            del myblock
        except:
            pass
        myblock, okblock = getblock(procblk, confl, blk_Common)  # .... add result to the dict


        if not okblock:
            continue
        my_process_block_dict[str(j)] = copy.deepcopy(myblock)


    for k in range(1, int(num_proc) + 1):  #go through all pending processes
        if str(k) not in my_process_block_dict.keys():
            my_process_block_dict[str(k)] = copy.deepcopy(blk_Common)

    return (my_process_block_dict)

def donothing():
    return()

def parse_geometry(mymctl):
    tk_height = mymctl.window.winfo_screenheight()
    tk_width  = mymctl.window.winfo_screenwidth()
    windypos = tk_height - mymctl.masterwin_y
    default_geometry =(f"{mymctl.masterwin_x}x{mymctl.masterwin_y}+{mymctl.master_window_x_pos}+{mymctl.master_window_y_pos}")  # size the window

    if 'my_geometry' not in globals() or my_geometry == "" or my_geometry == None or my_geometry == '1x1+0+0':
        printandfile(f"My_geometry fails initial test. Defaulting")
        return default_geometry

    try:
        wsize,sx_pos,sy_pos=my_geometry.split('+')
    except ValueError:
        return default_geometry
    x_pos = int(sx_pos)
    y_pos = int(sy_pos)
    #Check x_pos > monitor width
    if x_pos > tk_width:
        xpo = x_pos
        x_pos = x_pos - tk_width
    if y_pos > tk_height:
        ypo = y_pos
        y_pos = y_pos - tk_height

    new_geometry = f"{wsize}+{x_pos}+{y_pos}"
    return(new_geometry)





def master(num_processes, myprocdict, myevent, mctl):
    """
    Central hub used to distribute messages between processes, collect stats etc.
    """
    global do_update, showmaster, shutting_down
    showmaster = mctl.showmaster
    silence(mctl.hush)
    atexit.register(master_shutdown, mctl)
    setattr(mctl, 'procnum',-1)  #set procnum -1, because master
    setattr(mctl, 'devdict',get_device_dict())  #get GPU devices
    setattr(mctl, 'num_gpu', len(mctl.devdict)) #number of GPUs is number of device settings in devdict
    setattr(mctl, "event", myevent)  # event to use for video ready
    #setattr(mctl, "videoready_counter", num_processes)  # used to count down
    setattr(mctl, "initready_counter", num_processes)  # used to count down until init ready
    setattr(mctl, "shutdown_counter", num_processes)  # used to count down to monitor process shutdown
    setattr(mctl, "shutdown_action", None)  # placeholder, shutdown action. -1 = shutdown, +1 = resurrect
    setattr(mctl, "totallines", (num_processes + 2 + mctl.num_gpu + 5)) # total lines master display
    setattr(mctl, "topdataline", 0)  # position of topmost data line master display
    setattr(mctl, "datalinedict", {})  # holds the dataline label-objects in use
    setattr(mctl, "statuslinedict", {})  # holds the statusline label-objects in use
    setattr(mctl, "masterwin_x", 895)  # width master display
    setattr(mctl, "masterwin_y", (mctl.totallines + 2) * 20)  # height master display
    setattr(mctl, 'maxline', 82)  # maximum number of characters in master display line
    setattr(mctl, 'procdict', myprocdict) # store procdict
    setattr(mctl, 'centertext', mctl.masterwin_x / 2)
    setattr(mctl, 'lastseendict',{}) #
    setattr(mctl, 'initdict', {})
    setattr(mctl, 'statsdict', {})
    setattr(mctl, 'num_processes', num_processes)
    setattr(mctl, 'all_running', False)


    ################# for debug ####################################
    #print(f'Masterprocess - start {datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')
    printandfile(f'Master process - start')
    gc.enable()
    rtm = RepeatedTimer(300, writeobj,mctl)
    nfile = argv[0]
    nfile = nfile[:-3]+f"_objd_{mctl.procnum}" + ".log"
    with open(nfile, 'a+') as f:
        f.write("\n" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') +" STARTED" +"\n")
    writeobj(mctl) # do first one

    ################# for debug ####################################

    if showmaster:
        #put the master window where we left it - if we did ....

        setattr(mctl, 'window',[])
        # Master window setup
        setattr(mctl, 'window', tk.Tk()) # setup our master window
        # get monitor dimensions
        the_geometry = parse_geometry(mctl)
        mctl.window.geometry(the_geometry) # default or old gemometry
        mctl.window.geometry(f"{mctl.masterwin_x}x{mctl.masterwin_y}") # set width, height as needed
        mctl.window.title(mctl.master_window_title)
        mctl.window.protocol("WM_DELETE_WINDOW", functools.partial(master_shutdown, mctl))
        # menu
        menubar = Menu(mctl.window)
        runmenu = Menu(menubar, tearoff=0)
        runmenu.add_command(label="Quit" , background ='red', activebackground ='red3',  command=functools.partial(button_shutdown, mctl))
        runmenu.add_command(label="Restart", background ='green3', activebackground ='green', command=functools.partial(button_resurrect, mctl))
        menubar.add_cascade(label="Stop", menu=runmenu)

        recordmenu = Menu(menubar, tearoff=0)
        recordmenu.add_command(label="Record On", background ='green3',  activebackground ='green', command=functools.partial(button_record_all, mctl))
        recordmenu.add_command(label="Record Off", background ='red', activebackground ='red3', command=functools.partial(button_stoprecord_all, mctl))
        recordmenu.add_command(label="AutoRec On", background ='green3',  activebackground ='green', command=functools.partial(button_autovid_on, mctl))
        recordmenu.add_command(label="AutoRec Off", background ='red', activebackground ='red3', command=functools.partial(button_autovid_off, mctl))
        menubar.add_cascade(label="Record", menu=recordmenu)

        miscmenu = Menu(menubar, tearoff=0)
        miscmenu.add_command(label="Ding On", background ='green3',  activebackground ='green', command=functools.partial(button_audio_on, mctl))
        miscmenu.add_command(label="Ding Off", background ='red', activebackground ='red3', command=functools.partial(button_audio_off, mctl))
        miscmenu.add_command(label="OSD On", background ='green3',  activebackground ='green', command=functools.partial(button_osd_on, mctl))
        miscmenu.add_command(label="OSD Off", background ='red', activebackground ='red3', command=functools.partial(button_osd_off, mctl))
        miscmenu.add_command(label="Hush On", background ='green3',  activebackground ='green', command=functools.partial(button_hush_on, mctl))
        miscmenu.add_command(label="Hush Off", background ='red', activebackground ='red3', command=functools.partial(button_hush_off, mctl))
        miscmenu.add_command(label="Stats On", background ='green3',  activebackground ='green', command=functools.partial(button_stats_on, mctl))
        miscmenu.add_command(label="Stats Off", background ='red', activebackground ='red3', command=functools.partial(button_stats_off, mctl))
        miscmenu.add_command(label="Redraw", background ='yellow', activebackground ='yellow2', command=functools.partial(button_window_redraw, mctl))
        menubar.add_cascade(label="Misc", menu=miscmenu)

        mctl.window.config(menu=menubar)

        ml = tk.Label(mctl.window, text = "Starting up ... ", font = "Arial 24 bold")
        ml.pack(anchor = 'center')
        time.sleep(3)
        setattr(mctl, 'tabControl', ttk.Notebook(mctl.window, width=mctl.masterwin_x, height=mctl.masterwin_y))
        setattr(mctl, 'ltab', ttk.Frame(mctl.tabControl))
        setattr(mctl, 'etab', ttk.Frame(mctl.tabControl))
        setattr(mctl, 'stab', ttk.Frame(mctl.tabControl))
        mctl.tabControl.add(mctl.ltab, text ='Process'  )
        mctl.tabControl.add(mctl.etab, text ='Errors ')
        mctl.tabControl.add(mctl.stab, text ='Stats  ')
        ml.pack_forget()
        mctl.tabControl.pack(expand = True, fill= 'both')

        """
        btnframe = tk.Frame(mctl.stab)
        hi_there = tk.Button(btnframe,text="say hi")
        hi_there.pack()
        hi_again = tk.Button(btnframe,text="say hi again")
        hi_again.pack()
        btnframe.grid( row = 0, column = 0)
        """


        #####

        setrows3col(mctl.ltab,mctl.totallines) # set up the grid we'll be using

    if showmaster:
        settext(mctl.ltab,"Process Status", 1, 1, font = "Arial" , bold = True, size = 16)

        if mctl.has_config == 1:
            cket = settext(mctl.ltab,"Check error tab if processes won't go ready", 2 , 1, font = "Arial" , bold = False, size = 12)
        if mctl.has_config == -1:
            cket = settext(mctl.ltab,"No config file, working with defaults", 2, 1, font = "Arial" , bold = False, size = 12)
        if mctl.has_config == -2:
            cket = settext(mctl.ltab,"Config file empty, working with defaults", 2, 1, font = "Arial" , bold = False, size = 12)
        if mctl.has_config == -3:
            cket = settext(mctl.ltab,"Partial config file, working with defaults", 2, 1, font = "Arial" , bold = False, size = 12)
        cket_displayed = True


        msgline = "Stats will become available after video has started ...".ljust(mctl.maxline).rjust(mctl.maxline)
        settext(mctl.stab,msgline,2,1, bold = False, size = 14, font = "Arial")
        try:
            mctl.window.update()
        except:  # tkinter does not return proper error type
            pass

        setrows(mctl.etab,mctl.totallines) # set up the grid we'll be using
        setrows(mctl.stab,mctl.totallines) # set up the grid we'll be using

    rt = RepeatedTimer(0.5, update_master_window)
    #setattr(mctl,'rt', rt)
    m_accum_time = curr_cycle = 0
    m_prev_time = timer()
    try:
        while True:
            """
            This monitors the Master in-queue for commandsets sent by processes
            Commansets are validated on input, and discarded if invalid
            A commandset addressed to a specific video-process (To = x) is forwarded to that process
            A commandset addressed to all (To = 0) is forwarded to all video-processes except the sending
            A commandset addressed to Master (To = -1) is acted upon by the master process
            Slots < -1 are reserved for future expansion """  # pylint: disable=W0105

            if do_update and showmaster:
                try:
                    mctl.window.update()
                except:      #tkinter does not return proper error type
                    pass
                do_update = False

            assert(mctl != None)

            all_dead = True
            for key in mctl.procdict: #It's a dict

                if key == -1:
                    continue
                myset = mctl.procdict.get(key,{})
                if not get_proc_prop(mctl,key, 'is_running') and not shutting_down: # process declared dead
                    if not get_proc_prop(mctl,key, 'is_pronounced_dead'):
                        printandfile(f"Caught zombie {key}, dispatching ")
                        if showmaster:
                            do_statusline(mctl, key, ["Process shut down",'          ','ml          '])
                        mctl = put_proc_prop(mctl,key, 'is_pronounced_dead', True)
                        writethelog(f"{key} not is_running. Was not pronounced_dead. Now pronounced dead, and terminated")
                        get_proc_prop(mctl,key, 'process').terminate()
                else: #Process seems alive. Don't interrogate dead process

                    #if key > 0 and not get_proc_prop(mctl,key, 'process').is_alive():
                    #    #print(f"Proc {key} is dead")
                    #    continue

                    if key > 0 and get_proc_prop(mctl,key, 'process').is_alive(): #count only video processes
                        all_dead = False

                    ##Check queue
                    try:
                        my_commandset = get_proc_prop(mctl,key,'out_queue').get_nowait()
                        #my_commandset = myset.get('out_queue', None).get_nowait()  #'out_queue' is the in_queue here, query it
                    except queue.Empty:
                        continue
                    if not validate_commandset(my_commandset,
                                               num_processes):  # perform tests, returns True for good, False for bad
                        continue
                    # now we have a good commandset, operate on To (all parameters validated, no further checks necessary)
                    # check whether one process is signaling the others to stop recording. Set the Record button back to record,
                    # in case it was set to stop recording
                    if my_commandset != {}:
                        if my_commandset['Command'].strip() == "stoprecording":
                            stoprecord_all(mctl, doqueue = False)
                        # same for startrecording
                        if my_commandset['Command'].strip() == "startrecording":
                            record_all(mctl, doqueue = False)
                        mctl = process_master_commandset(mctl, my_commandset)
                        assert(mctl != None)

            if all_dead and not shutting_down: #shut down master if all video processes are dead, but don't interrupt a shutdopwn/restart in progress
                printandfile("All video processes dead. Shutting down.")
                if mctl.all_dead_restart: #restart, don't shutdown
                    master_shutdown(mctl, why = "All dead", restart = True)
                else:
                   master_shutdown(mctl, why = "All dead", restart = False)

            #loop timimg
            m_curr_time = timer()
            m_exec_time = m_curr_time - m_prev_time
            m_prev_time = m_curr_time
            m_accum_time = m_accum_time + m_exec_time
            curr_cycle += 1

            if m_accum_time > 1:  #happens each second
                #patrol any outstanding sync-starts

                if mctl.sync_start and is_video_ready(mctl):
                    # Video ready, patrol...
                    mctl.event.set()
                    mctl.sync_start = False


                mctl = check_all_running(mctl)

                if mctl.redraw and mctl.all_running: #issue redraw command if needed
                    for _ in range(0,3):
                        send_to_all({ 'To' : 0, 'From' :-1,  'Command' : "windowredraw", 'Args':{}}, mctl.procdict)
                        time.sleep(0.5)
                    mctl.redraw = False

                if cket.winfo_exists() and mctl.all_running: #Remove "Check error tab if ..." when all active
                    cket.destroy()

                m_accum_time -= 1
                curr_cycle = 0
                assert(mctl != None)
                if not shutting_down:
                    mctl = check_for_hung_process(mctl) #patrol for hung processes
                    assert(mctl != None and mctl.procdict != None)
            #end loop timing
            time.sleep(0.005)  # waste a little time
    except KeyboardInterrupt as ki:
        printandfile(f"Master process caught Ctl-c - shutting down")
        master_shutdown(mctl) # Shutdown

    sys.exit()

def update_master_window():
    global do_update
    do_update = True
    return()


def process_master_commandset(mctlx, my_commandset):
    """
    This distributes commands queued to master to their final destination(s)
    Command sets sent to master will stay in master, all others are passed on to their intended processes
    """
    sendto = my_commandset.get('To', None)
    assert sendto != None, f"Failure in Master recipient logic @ line {sys._getframe().f_lineno}. No 'To:' found."
    if sendto  == 0:  # send the received commandset out to all processes, except the one that sent it
        send_to_all(my_commandset, mctlx.procdict)
        return(mctlx)
    if sendto  > 0 or sendto  < -1:  # Single recipient, put into proper queue
        get_proc_prop(mctlx, sendto,'out_queue').put(my_commandset)
        return(mctlx)
    # at this point, it must be addressed to master (< 0) but check to make sure
    assert my_commandset['To'] < 0, f"Failure in Master recipient logic @ line {sys._getframe().f_lineno}"
    mctlx = process_master_commands(my_commandset, mctlx)
    return(mctlx)


def validate_commandset(my_commandset, num_processes):
    """
    Perform sanity checks on commandset, return True if good, False if bad
    """
    if not isinstance(my_commandset,dict):
        printandfile(f'Commandset {my_commandset} must be dict, is {type(my_commandset)} @ line {sys._getframe().f_lineno}') # pylint: disable=W0212
        return (False)
    try:
        if not isinstance(my_commandset['To'], int) or not isinstance(my_commandset['From'], int):
            printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno} check To and From') # pylint: disable=W0212
            return (False)
    except KeyError:
        printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno} check To and From, key failed') # pylint: disable=W0212
        return (False)
    if my_commandset['To'] > num_processes or my_commandset['From'] > num_processes:
        printandfile(f"Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, To {my_commandset['To']} or From {my_commandset['From']} out of range") # pylint: disable=W0212
        return (False)
    try:
        if not isinstance(my_commandset['Command'], str)  or my_commandset['Command'] == "":
            printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Command') # pylint: disable=W0212
            return (False)
    except KeyError:
        printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Command, key failed') # pylint: disable=W0212
        return (False)
    if not my_commandset['Command'].isalpha():
        printandfile(f"Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, Command {my_commandset['Command']} must be letters only") # pylint: disable=W0212
        return (False)
    if 'Args' in my_commandset:
        try:
            if not isinstance(my_commandset['Args'],dict):
                printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, check Args, must be dict') # pylint: disable=W0212
                return (False)
        except KeyError:
            printandfile(f'Bad commandset {my_commandset} @ line {sys._getframe().f_lineno}, Args failed test') # pylint: disable=W0212
            return (False)
    return (True)

def get_global_active(my_mctl):
    """
    Signals that the majority of processes has gone active
    """
    active_procs = 0
    for key in my_mctl.procdict:
        if get_proc_prop(my_mctl,key, 'is_active'):
            active_procs += 1
    if active_procs >= math.floor((len(my_mctl.procdict) -1)*0.9):
        return(True)
    else:
        return(False)

def check_all_running(my_mctl):
    """
    If all appear to be running set a timer that announces 5 sec later that all sould be up
    """
    if isinstance(my_mctl.ost,threading.Timer): #timer already set, do nothing
        return(my_mctl)
    if len(my_mctl.lastseendict) == my_mctl.num_processes - get_tot_proc_prop(my_mctl, 'is_pronounced_dead'):
        my_mctl.ost = threading.Timer(10, all_running_timeout, [my_mctl,]) # set a timer for YOLO to initialize
        my_mctl.ost.start()
    return(my_mctl)


def get_tot_proc_prop(my_mctl, the_prop ):
    """return the total of the_prop set to True in my_mctl.procdict"""
    mycounter = 0
    for key in my_mctl.procdict:
        if key == -1:
            continue
        if get_proc_prop(my_mctl, key, the_prop):
            mycounter += 1
    return(mycounter)



def get_proc_prop(my_mctl,the_proc, the_prop):
    """
    Return the value of the_prop from my_mctl.procdict indexed by theproc
    For instance, get_proc_prop(my_mctl,2, 'is_running')  would retrieve the is_running state of process 2
    """
    try:
        assert(the_proc <= my_mctl.num_processes and the_proc >= -1)
        assert(my_mctl != None)
    except AssertionError:
        printandfile(f"Assertion error in get_proc_prop @ line {sys._getframe().f_lineno -2}. The_proc: {the_proc}. The_prop: {the_prop}. Type my_mctl: {type(my_mctl)}")
    try:
        the_val = my_mctl.procdict[the_proc][the_prop]
    except KeyError as e:
        printandfile(f'KeyError @ line {sys._getframe().f_lineno -2}, process {the_proc}') # pylint: disable=W021
        sys.exit()
    return(the_val)


def put_proc_prop(my_mctl,the_proc, the_prop, the_val):
    """
    Set the value of the_prop from my_mctl.procdict indexed by theproc to the_val
    For instance, put_proc_prop(my_mctl,2, 'is_running',True)  would set the is_running state of process 2 to True
    Returns my_mctl with updated my_mctl.procdict
    """

    if the_prop in my_mctl.procdict[the_proc]:
        try:
            my_mctl.procdict[the_proc][the_prop] = the_val
        except KeyError as e:
            printandfile(f'{e} @ line {sys._getframe().f_lineno -2}, process {the_proc}') # pylint: disable=W021
            sys.exit()
    else:
        raise KeyError(f'KeyError @ line {sys._getframe().f_lineno -5}, unknown property {the_prop}') # pylint: disable=W021
        sys.exit()
    return(my_mctl)


def send_to_all(my_commandset, myprocdict):
    """
    Send same message to all processes in procdict
    """
    for key in myprocdict: #
        myset = myprocdict.get(key,{})
        if myset.get('process_number', -999) != my_commandset['From'] and myset.get('is_running', False):  # Do not send back to sender. Don't send to dead process
            myset.get('in_queue',None).put(my_commandset) # Put into in_queue of the process
    return()


def restart_process(myproc,mymctl):
    """ Try restarting a (supposedly hung) process"""
    writethelog(f'Master: Attempting to restart process {myproc}')
    #we need the config settings for that process
    _ , blk_process_block_dict = makeconfig()

    #get our block
    blk_process_block = blk_process_block_dict.get(str(myproc), None)
    if blk_process_block == None:
        printandfile(f'Master: blk_process_block_dict {myproc} returns None @line {sys._getframe().f_lineno -2}. Will not restart process.')
        return(mymctl)
    #set sync_start to False, we aren't weaiting for anybody
    setattr(blk_process_block,'sync_start', False)
    setattr(blk_process_block,'monitor_YOLO', False)
    setattr(blk_process_block,'hush', False)




    #just to make sure that process is definitely dead
    get_proc_prop(mymctl,myproc,'process').terminate()

    #set procdict to initial values
    mymctl = put_proc_prop(mymctl,myproc, 'is_videoready', False)
    mymctl = put_proc_prop(mymctl,myproc, 'is_active', False)
    mymctl = put_proc_prop(mymctl,myproc, 'is_pronounced_dead', False)
    mymctl = put_proc_prop(mymctl,myproc, 'is_running', True)
    #the  multiprocessing.Event() won't be used, but we need the type

    """
    print(dir(blk_process_block))
    print(blk_process_block.model_path)
    print(blk_process_block.anchors_path)
    print(blk_process_block.classes_path)
    """

    myprocess = multiprocessing.Process(target=video_process, name=f'VP{myproc}', args=(get_proc_prop(mymctl,myproc,'in_queue'),get_proc_prop(mymctl,myproc,'out_queue'), blk_process_block, multiprocessing.Event(), myproc ))
    myprocess.start()
    #and store the process for later
    mymctl = put_proc_prop(mymctl,myproc, 'process', myprocess)
    return(mymctl)








def dump_garbage():
    """
    show us what's the garbage about
    """

    # force collection
    print("\nGARBAGE:")
    gc.collect()

    print("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80: s = s[:80]
        print(type(x),"\n  ", s)


    print("=============================================")

    tracker.print_diff()

    print("==============================================")

    logging.debug(mem_top())

    return()

def main_killer(my_mctl):
    for key in my_mctl.procdict: #It's a dict
        if key == -1 :
            continue
        else:
            writethelog(f"Master: Main killer killed proc {key}")
            get_proc_prop(my_mctl,key,'process').terminate()
            my_mctl = put_proc_prop(my_mctl,key, 'is_active', False)
            my_mctl = put_proc_prop(my_mctl,key, 'is_running', False)
    return(my_mctl)



if __name__ == "__main__":
    """
    Main routine.
    Sets up Master and Video processes.
    Then gets out of the way, except for monitoring that processes have started alright.
    If a video process won't start, or if it dies, there will be a notification on the master screen, and the
    rest of the processes will continue.
    If the master process won't start, or if it dies, execution will be halted.
    """  # pylint: disable=W0105

    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)
    # tracker = SummaryTracker()

    fp = open(pid_file, 'w')
    # Check for running instance
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("Cannot start. Another instance running.")
        sys.exit(0)
    atexit.register(startup_shutdown)  # orderly shutdown
    printandfile("\nMultiDetect start")

    blk_Master, blk_process_block_dict = makeconfig()
    # blk_Master has the settings for the Master process
    # blk_process_block_dict has one set of settings for each process, like {"1": FLAGS} and so forth
    assert blk_Master.num_processes > 0, "No suitable num_processes in Master block of config file. Aborting"

    m_in_queue = multiprocessing.Queue()
    m_out_queue = multiprocessing.Queue()
    processes = blk_Master.num_processes
    procdict = {}
    mpe = multiprocessing.Event()

    for pc in range(1, processes + 1):
        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()
        myprocess = multiprocessing.Process(target=video_process, name=f'VP{pc}', args=(in_queue, out_queue, blk_process_block_dict[str(pc)],mpe, pc ))
        procdict[pc] = {'in_queue' :in_queue, 'out_queue':out_queue, 'process_number': pc, 'process': myprocess, 'process_name' : f'VP{pc}', 'is_running':  True, 'is_pronounced_dead': False, 'is_active': False, 'is_videoready': False}
        myprocess.start()

    #procdict[-1] = [None,  None,  -1,  None, 'VPMaster',  True,  4 ] #add a record for communication between master and main
    procdict[-1] = {'in_queue': None, 'out_queue': None, 'process_number': -1, 'process': None, 'process_name' : 'VPMaster', 'is_running':  True, 'is_pronounced_dead': False, 'is_active': False, 'is_videoready': False}
    master(pc, procdict, mpe, blk_Master)
    sys.exit()
