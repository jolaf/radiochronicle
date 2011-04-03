#!/usr/bin/python
#
# RCProcess.py
#
# A part of RadioChronicle project
#
# by Vladimir Yashunsky (vladimir.yashunsky@gmail.com)
# and Vasily Zakharov (vmzakhar@gmail.com)
# http://code.google.com/p/radiochronicle/
#
# Version 0.1
#

from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime, timedelta, min as DATETIME_MIN, max as DATETIME_MAX
from getopt import getopt
from glob import glob
from inspect import getmembers
from logging import getLogger, StreamHandler, NOTSET
from logging.config import fileConfig
from re import sub
from signal import signal, SIGTERM
from struct import unpack
from sys import argv, exit # exit redefined # pylint: disable=W0622
from thread import start_new_thread
from time import sleep, strftime
from traceback import format_exc
import audioop
import wave

TITLE = "RCProcess v0.1  http://code.google.com/p/radiochronicle"
DEFAULT_CONFIG_FILE_NAME = 'rcp.conf'

def parseHTMLColor(htmlColor):
    '''Parses HTML #rrggbb color.'''
    htmlColor = htmlColor.strip()
    if htmlColor[0] == '#':
        htmlColor = htmlColor[1:]
    if len(htmlColor) != 6:
        raise ValueError("Bad HTML color format, expected #rrggbb")
    return (int(x, 16) for x in (htmlColor[:2], htmlColor[2:4], htmlColor[4:]))

def strftime(timeFormat, time):
    '''Formats time to a string, supporting additional format %: which is formatted
       to a ':' in the first half of a second, and to a ' ' in the second half.'''
    return time.strftime(timeFormat.replace('%:', ':' if time.microsecond < 500000 else ' '))

def strptime(timeFormat, time):
    '''Parses a time string, supporting additional format %:
       which stands for either ':' or ' ' in the input.'''
    try:
        return datetime.strptime(time, timeFormat.replace('%:', ':'))
    except ValueError:
        return datetime.strptime(time, timeFormat.replace('%:', ' '))

def validateTimeFormat(timeFormat):
    '''Returns True if the specified time format is valid, False otherwise.'''
    try:
        strftime(timeFormat, datetime.today())
        return True
    except ValueError:
        return False

def validateTime(timeFormat, time):
    '''Returns True if the specified time is correctly formatted
       according to the specified format, False otherwise.'''
    try:
        strptime(timeFormat, time)
        return True
    except ValueError:
        return False

def diffTime(startTime, endTime):
    '''Reverse of relativeTime().
       Removes the common beginning from the endTime.'''
    nd = 0
    for i in xrange(min(len(startTime), len(endTime))):
        if not endTime[i].isdigit():
            if startTime[:i + 1] == endTime[:i + 1]:
                nd = i + 1
            else:
                break
    return endTime[nd:]

def relativeTime(startTime, endTime):
    '''Reverse of diffTime().
       Left-pads endTime to the length of startTime using startTime's beginning.'''
    assert len(startTime) >= len(endTime)
    return startTime[:len(startTime) - len(endTime)] + endTime

def InheritConfigParser(ConfigParser):
    '''Simplifies handling of typed config parameters and handles section inheritance.'''

    def getString(self, section, option, raw = False, vars = None):
        '''Retrieves a string value from the particular section.'''
        return self.get(section, option, raw, vars).strip()

    def getInt(self, section, option, raw = False, vars = None):
        '''Retrieves an integer value from the particular section.'''
        try:
            value = self.getString(section, option, raw, vars)
            return int(value)
        except ValueError:
            raise ValueError("Bad value for [%s].%s: '%s', must be an integer" % (section, option, value))

    def getFloat(self, section, option, raw = False, vars = None):
        '''Retrieves a float value from the particular section.'''
        try:
            value = self.getString(section, option, raw, vars)
            return float(value)
        except ValueError:
            raise ValueError("Bad value for [%s].%s: '%s', must be a float" % (section, option, value))

    def getBoolean(self, section, option):
        '''Retrieves a boolean value from the particular section.'''
        try:
            return self.getboolean(section, option)
        except ValueError:
            raise ValueError("Bad value for [%s].%s: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, option, self.getString(section, option)))

    def getSections(self, section, previousSections = []): # generator # mutable default is ok # pylint: disable=W0102
        '''Resursively retrieves list of sections considering inheritance.'''
        if not self.has_section(section):
            raise Exception("Section [%s] not found" % section)
        yield section
        sections = [section]
        ps = previousSections + sections
        if self.has_option(section, 'inherit'):
            for s in self.get(section, 'inherit').split():
                if s in ps:
                    raise ValueError("Inheritance recursion detected: %s -> %s" % (section, s))
                for section in self.getSections(s, ps):
                    yield section

    def _getValue(self, section, option, default, raw = False, vars = None)
        t = type(default)
        if t == bool:
            return self.getBoolean(section, option)
        if t == float:
            return self.getFloat(section, option, raw, vars)
        if t == int:
            return self.getInt(section, option, raw, vars)
        if t == str:
            return self.getString(section, option, raw, vars)
        return t(self.getString(section, option, raw, vars))

    def getValue(self, section, option, default, raw = False, vars = None):
        '''Retrieves a value, considering section inheritance and using default's type.'''
        for section in self.getSections(section):
            if self.has_option(section, option):
                return self._getValue(section, option, default, raw = False, vars = None)
        return default

    def getValues(self, section, defaults, allowUnknown = False, returnDefaults = False, raw = False, vars = None): # generator
        '''Retrieves the specified values, considering section inheritance and using defaults' types.'''
        unset = dict(defaults)
        for section in self.getSections(section):
            for option in self.options(section):
                for option in (default for default in defaults if default.lower() == option): # single iteration only
                    if option in unset:
                        yield (option, self._getValue(section, option, defaults[option], raw = False, vars = None))
                        del unset[option]
                        if not unset and allowUnknown:
                            return
                    break
                elif not allowUnknown:
                    raise ValueError("Unknown option [%s].%s" % (section, option))
        if returnDefaults:
            for (option, value) in unset.iteritems():
                yield (option, value)

class Configurable(object):
    '''Describes an object configurable using an InheritConfigParser section.'''

    def __init__(self, config, section):
        self.config = config
        self.section = section
        defaults = dict((name, value) for (name, value) in getmembers(self, lambda x: not callable(x)) if name[:1].islower())
        for (field, value) in config.getValues(section, defaults):
            setattr(self, field, value)

class TimePoint(object):
    START = 'START'	# If timepoint is inside the file, start from the beginning of that file
    END = 'END'		# If timepoint is inside the file, end by the end of that file
    NEXT = 'NEXT'	# If timepoint is inside the file, start from the beginning of the next file
    PREV = 'PREV'	# If timepoint is inside the file, end by the end of the previous file
    CUT = 'CUT'		# If timepoint is inside the file, start from (end by) exactly that point
    SKIP = 'SKIP'	# If timepoint is outside any file, start from the beginning (end by the end) of the next (previous) file
    KEEP = 'KEEP'	# If timepoint is outside any file, start from (end by) exactly that point (keep silence)

    time = None
    fromStart = toEnd = True	# If time is set, exactly one of fromStart/toEnd, fromNext/toPrev and cut must be True
    fromNext = toPrev = False
    cut = False		
    keep = False

    def __init__(self, timeFormat, s, location = ''): # ToDo: do we need time format here?
        if type(s) == datetime:
            self.time = s
            return
        args = s.split()
        if not args:
            raise ValueError("Date not specified%s" % ((' at %s' % location) if location else ''))
        time = args[0]
        try:
            self.time = strptime(timeFormat, time)
        except ValueError, e:
            raise ValueError("Format '%s' is not matched by the data%s: '%s'" % (timeFormat, (' at %s' % location) if location else '', time))
        for arg in (arg.upper() for arg in args[1:]):
            # ToDo: Make sure only one specifier of each kind exists?
            if arg in (self.START, self.END, self.NEXT, self.PREV, self.CUT):
                # ToDo: Add separate checks for Start and End points
                self.fromStart = self.toEnd = arg in (self.START, self.END)
                self.fromNext = self.toPrev = arg in (self.NEXT, self.PREV)
                self.cut = (arg == self.CUT)
            elif arg in (self.SKIP, self.KEEP):
                self.keep = (arg == self.KEEP)
            else:
                raise ValueError("Unknown time point specifier%s: '%s'" % ((' at %s' % location) if location else '', arg))

    def __cmp__(self, other):
        return self.time.__cmp__(other.time)

    def matchStart(self, startTime, endTime):
        # ToDo

    def matchEnd(self, startTime, endTime):
        # ToDo

class InputFile(object):
    '''Holds information on the input audio file and methods to access it.'''
    N_CHANNELS_NAMES = { 1: 'Mono', 2: 'Stereo' }

    # ToDo: Change Audio library to support mp3 etc.
    def __init__(self, fileName, startTime):
        self.fileName = fileName
        self.startTime = startTime
        self.wave = wave.open(fileName)
        (self.nChannels, self.audioBytes, self.sampleRate, self.nSamples, compressionType, compressionName) = self.wave.getparams() # compressionName is unused
        if compressionType != 'NONE':
            raise ValueError("Unknown compression type '%s' for file %s" % (compressionType, fileName))
        self.audioBits = self.audioBytes * 8
        self.sampleSize = self.nChannels * self.audioBytes
        self.length = timedelta(seconds = float(self.nSamples) / self.sampleRate)
        self.endTime = self.startTime + self.length
        # ToDo: Maybe round length right in the fields?
        self.printLength = self.length + timedelta(microseconds = (self.length.microseconds + 5000) // 10000 * 10000 - self.length.microseconds)
        self.name = "%s %dHz/%d-bit/%s %s" % (self.fileName, self.sampleRate, self.audioBits, self.N_CHANNELS_NAMES.get(self.nChannels) or '%d-channels' % self.nChannels, self.printLength)

    def contains(self, time):
        return self.startTime <= time < self.endTime

    def __str__(self):
        return self.name

    def __repr__(self):
        return "InputFile(%s, %s)" % (repr(self.fileName), repr(self.startTime))

    def read(self, nSamples):
        ret = self.wave.readframes(nSamples)
        assert len(ret) == nSamples * self.sampleSize
        return ret

    def rewind(self):
        self.wave.rewind()

    def close(self):
        self.wave.close()

class Input(Configurable):
    '''Describes a time range and a set of input files.'''

    NOT_SET = 0
    FIRST_FILE = LAST_FILE = 1
    NEXT_FILE = PREVIOUS_FILE = 2
    CUT_FILE = 3

    STEREO = 0
    LEFT = 1
    RIGHT = 2
    MIX = 3

    CHANNEL_NUMBERS = { 'STEREO': STEREO, 'LEFT': LEFT, 'RIGHT': RIGHT, 'MIX': MIX }
    CHANNEL_NAMES = { STEREO: 'STEREO', LEFT: 'LEFT', RIGHT: 'RIGHT', MIX: 'MIX' }

    fileNameFormat = './RC-%Y%m%d-%H%M%S.wav' # ToDo: define through general setting
    maxFiles = 0
    startPoint = ''
    endPoint = ''
    channels = ''

    def __init__(self, config, section):
        Configurable.__init__(self, config, section)
        if not validateTimeFormat(fileNameFormat):
            raise ValueError("Bad file name format at [%s].fileNameFormat: '%s'" % (section, self.fileNameFormat))
        self.startPoint = TimePoint(self.startPoint, '[%s].startPoint' % section) if self.startPoint else TimePoint(DATETIME_MIN)
        self.endPoint = TimePoint(self.endPoint, '[%s].endPoint' % section) if self.endPoint else TimePoint(DATETIME_MAX)
        if self.channels:
            try:
                self.channels = self.CHANNEL_NUMBERS[self.channels.upper()]
            except KeyError:
                raise ValueError("Bad channels specification at [%s].channels: '%s'" % (section, self.channels))
        else:
            self.channels = STEREO

    @staticmethod
    def getFileTimes(fileNameFormat): # generator
        for fileName in glob(sub('%.', '*', fileNameFormat)):
            try:
                yield (strptime(fileNameFormat, fileName), fileName)
            except ValueError:
                pass

    @staticmethod
    def preSortFiles(files, startTime, endTime): # generator
        assert startTime <= endTime
        ret = []
        started = False
        prev = None
        for (time, fileName) in sorted(files):
            if not started:
                if time >= startTime:
                    if time > startTime and prev:
                        yield prev
                    started = True
                else:
                    prev = (time, fileName)
            if started:
                if time <= endTime:
                    yield (time, fileName)
                else:
                    return

    @staticmethod
    def openFiles(files): # generator
        prev = None
        self.startTime = self.endTime = None
        for (time, fileName) in files:
            f = InputFile(fileName, time)
            if prev and f.startTime < prev.endTime:
                raise ValueError("Files intersect in time: %s start < %s end" % (f, prev))
            prev = f
            if not self.startTime:
                self.startTime = self.startPoint.matchStart(f.startTime, f.endTime)
                if not self.startTime or f.endTime < self.startTime:
                    continue
            if not self.endTime:
                self.endTime = self.startPoint.matchEnd(f.startTime, f.endTime)
            if f.endTime > self.startTime and not self.endTime
                yield f

    def load(self):
        self.files = tuple(self.openFiles(self.preSortFiles(self.getFileNames(self.fileNameFormat), self.startPoint.time, self.endPoint.time)))
        if not self.files:
            raise ValueError("No files found matching [%s].fileNameFormat: '%s'" % (section, self.fileNameFormat))

class RCProcess:
    # Default parameter values
    logger = None

    def __init__(self):
        '''Fully constructs class instance, including reading configuration file and configuring audio devices.'''
        try: # Reading command line options
            configFileName = DEFAULT_CONFIG_FILE_NAME
            (options, args) = getopt(argv[1:], 'c:h', ('config=', 'help')) # args in unused # pylint: disable=W0612
            for (option, value) in options:
                if option in ('-c', '--config'):
                    configFileName = value.strip()
                else:
                    usage()
        except Exception, e:
            usage("Error: %s\n" % e)
        try: # Reading config file and configuring logging
            config = ConfigParser()
            config.readfp(open(configFileName)) # Using readfp(open()) to make sure file exists
            if config.has_section('loggers'):
                fileConfig(configFileName)
            self.logger = getLogger()
            if not self.logger.handlers: # Provide default logger
                self.logger.addHandler(StreamHandler())
                self.logger.setLevel(NOTSET)
            signal(SIGTERM, self.sigTerm)
        except Exception, e:
            print "%s\n\nConfig error: %s" % (TITLE, e)
            exit(1)
        # Above this point, use print for diagnostics
        # From this point on, we have self.logger to use instead
        self.logger.info(TITLE)
        self.logger.info("Using %s" % configFileName)
        print # Empty line to console only
        try: # Applying configuration
            self.inputs = tuple(Input(config, section) for section in sorted(section for section in config.sections() if section.startswith('input')))
            if not self.inputs:
                raise Exception("No [input*] sections found")
            try:
                # ToDo: create wrappers for getting config parameters, for RC also
                section = 'general'
                try:
                    self.fileNameFormat = config.get(section, 'fileNameFormat').strip()
                    # ToDo: check that files exist
                except NoOptionError: pass
                try:
                    self.startDate = config.get(section, 'startDate')
                    # ToDo: parse date
                except NoOptionError: pass
                try:
                    self.endDate = config.get(section, 'endDate')
                    # ToDo: parse date
                except NoOptionError: pass
                try:
                    self.video = config.getboolean(section, 'video')
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].video: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, config.get(section, 'video')))
                try:
                    self.audio = config.getboolean(section, 'audio')
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].audio: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, config.get(section, 'audio')))
            except NoSectionError: pass
            try:
                section = 'video'
                # ToDo: specify video format somehow
                try:
                    value = config.get(section, 'width')
                    self.width = int(value)
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].width: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'height')
                    self.height = int(value)
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].height: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'fps')
                    self.fps = float(value)
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].fps: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'background')
                    self.background = parseHTMLColor(value)
                    # ToDo: think about proper color representation
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].background: '%s', must be a #rrggbb string" % (section, value))
                try:
                    self.previewVideo = config.getboolean(section, 'preview')
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].preview: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, config.get(section, 'preview')))
            except NoSectionError: pass
            try:
                section = 'audio'
                try:
                try:
                    value = config.get(section, 'audioBits')
                    self.audioBits = int(value)
                    # ToDo: check it further
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].audioBits: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'sampleRate')
                    self.sampleRate = int(value)
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].sampleRate: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'channels')
                    self.channels = int(value)
                    # ToDo: check it's positive
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].channels: '%s', must be an integer" % (section, value))
                try:
                    self.previewAudio = config.getboolean(section, 'preview')
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].preview: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, config.get(section, 'preview')))
            except NoSectionError: pass
            try:
                section = 'tuning'
                try:
                    value = config.get(section, 'maxPauseLength')
                    self.maxPauseLength = float(value)
                    self.replacePauseLength = self.maxPauseLength
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].maxPauseLength: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'replacePauseLength')
                    self.replacePauseLength = float(value)
                    # ToDo: check replacePauseLength <= maxPauseLength
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].replacePauseLength: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'trailLength')
                    self.trailLength = float(value)
                except NoOptionError: pass
                except ValueError:
                    raise ValueError("Bad value for [%s].trailLength: '%s', must be a float" % (section, value))
            except NoSectionError: pass

            # Validating configuration parameters
            if not self.fileNameFormat:
                raise ValueError("Bad value for fileNameFormat: must be not empty")
            if not 0 <= self.volumeTreshold <= 100:
                raise ValueError("Bad value for volumeTreshold: %.2f, must be 0-100" % self.volumeTreshold)
            if self.maxPauseLength < 0:
                self.maxPauseLength = 0
            if self.minRecordingLength < 0:
                self.minRecordingLength = 0
            if self.trailLength < 0:
                self.trailLength = 0
            if self.chunkSize < 1:
                raise ValueError("Bad value for chunkSize: %d, must be 1 or more" % self.chunkSize)
            if self.inputDevice:
                if self.inputDevice == -1:
                    self.inputDevice = None
                elif self.inputDevice < -1:
                    raise ValueError("Bad value for input device: %d, must be -1 or more" % self.inputDevice)
            if self.outputDevice:
                if self.outputDevice == -1:
                    self.outputDevice = None
                elif self.outputDevice < -1:
                    raise ValueError("Bad value for output device: %d, must be -1 or more" % self.outputDevice)
            if self.audioBits not in (8, 16, 32):
                raise ValueError("Bad value for audioBits: %d, must be 8, 16, or 32" % self.audioBits)
            if self.sampleRate < 1:
                raise ValueError("Bad value for chunkSize: %d, must be positive" % self.sampleRate)
            try:
                self.channel = int(channel)
                if self.channel <= 0:
                    self.channel = None # Exception will be thrown below
            except ValueError:
                self.channel = CHANNEL_NUMBERS.get(channel.strip().upper()) # Would be None if not found
            if self.channel == None:
                raise ValueError("Bad value for channel: %s, must be LEFT/RIGHT/STEREO/ALL/MONO or a number of 1 or more" % channel)

            # Accessing PyAudio engine
            try:
                from pyaudio import PyAudio
            except ImportError, e:
                raise ImportError("%s: %s\nPlease install PyAudio: http://people.csail.mit.edu/hubert/pyaudio" % (e.__class__.__name__, e))
            self.audio = PyAudio()
            print "%s\n" % self.deviceInfo() # Using print for non-functional logging

            # Accessing audio devices
            try:
                if self.inputDevice != None:
                    inputDeviceInfo = self.audio.get_device_info_by_index(self.inputDevice)
                    self.logger.info("Using input device %s" % self.deviceInfo(inputDeviceInfo, False))
                else:
                    inputDeviceInfo = self.audio.get_default_input_device_info()
                    self.logger.info("Using default input device %s" % self.deviceInfo(inputDeviceInfo, False))
            except ValueError:
                raise ValueError("%s is not in fact an input device" % ("Input device %d" % self.inputDevice if self.inputDevice != None else "Default input device"))
            except IOError, e:
                raise IOError("Can't access %s: %s" % ("input device %d" % self.inputDevice if self.inputDevice != None else "default input device", e))
            try:
                if self.outputDevice != None:
                    outputDeviceInfo = self.audio.get_device_info_by_index(self.outputDevice)
                    self.logger.info("Using output device %s" % self.deviceInfo(outputDeviceInfo, True))
                else:
                    outputDeviceInfo = self.audio.get_default_output_device_info()
                    self.logger.info("Using default output device %s" % self.deviceInfo(outputDeviceInfo, True))
            except ValueError:
                raise ValueError("%s is not in fact an output device" % ("output device %d" % self.outputDevice if self.outputDevice != None else "Default output device"))
            except IOError, e:
                raise IOError("Can't access %s: %s" % ("output device %d" % self.outputDevice if self.outputDevice != None else "default output device", e))
            print # Empty line to console only

            # Calculating derivative paratemers
            self.numInputChannels = 1 if self.channel == MONO else inputDeviceInfo['maxInputChannels']
            assert self.numInputChannels > 0
            if self.channel > self.numInputChannels:
                raise ValueError("Bad value for channel: %d, must be no more than %d" % (self.channel, self.numInputChannels))
            self.numOutputChannels = self.numInputChannels if self.channel == STEREO else 1
            assert self.numOutputChannels > 0

            self.audioBytes = self.audioBits // 8
            self.maxVolume = 1 << (self.audioBits - 1)
            self.audioFormat = self.audio.get_format_from_width(self.audioBytes, False)
            self.packFormat = PACK_FORMATS[self.audioBits]

            self.inputBlockSize = self.numInputChannels * self.chunkSize * self.audioBytes
            self.outputBlockSize = self.numOutputChannels * self.chunkSize * self.audioBytes
            self.inputSecondSize = self.numInputChannels * self.sampleRate * self.audioBytes
            self.outputSecondSize = self.numOutputChannels * self.sampleRate * self.audioBytes
            self.chunksInSecond = self.sampleRate // self.chunkSize
            self.chunksToStop = self.chunksInSecond * self.maxPauseLength
            self.chunksOfFadeout = self.chunksInSecond * self.trailLength

            # Diagnosting audio devices
            if not self.createInputStream():
                raise Exception("Can't create input stream")
            self.closeInputStream()
            if not self.createOutputStream():
                raise Exception("Can't create output stream")
            self.closeOutputStream()

            # Printing configuration info
            self.logger.info("Recording %dHz/%d-bit/%s to %s" % (self.sampleRate, self.audioBits, CHANNEL_NAMES.get(self.channel) or "channel %d" % self.channel, self.fileNameFormat))
            self.logger.info("Volume threshold %.2f%%, max pause %.1f seconds, min recording length %.1f seconds, trail %.1f seconds" % (self.volumeTreshold, self.maxPauseLength, self.minRecordingLength, self.trailLength))
            self.logger.info("Monitor is %s" % ('ON' if self.monitor else 'OFF'))
            print "Type 'help' for console commands reference" # Using print for non-functional logging
            print # Empty line to console only
        except Exception, e:
            self.logger.error("Configuration error: %s" % e)
            exit(1)

    def run(self):
        '''Runs main audio processing loop.'''
        self.audioFile = None
        self.sampleLength = 0
        self.audioFileLength = 0
        self.inLoop = True
        self.recording = False
        self.quitAfterRecording = False
        self.lastSecondVolumes = [0] * self.chunksInSecond
        chunkInSecond = 0
        start_new_thread(self.commandConsole, ()) # Start command console thread
        self.logger.info("Listening started")

        # Main audio processing loop
        try:
            while self.inLoop:
                # Retrieve next chunk of audio data
                data = self.readAudioData()
                if not data: # Error occurred
                    sleep(1.0 / self.chunksInSecond) # Avoid querying malfunctioning device too often
                    continue
                assert len(data) == self.inputBlockSize

                if self.channel not in (MONO, STEREO): # Extract the data for particular channel
                    data = ''.join(data[i : i + self.audioBytes] for i in xrange((self.channel - 1) * self.audioBytes, len(data), self.numInputChannels * self.audioBytes))
                assert len(data) == self.outputBlockSize

                if self.monitor: # Provide monitor output
                    self.writeAudioData(data)

                # Gathering volume statistics
                volume = (mean(abs(unpack(self.packFormat, data[i : i + self.audioBytes])[0]) for i in xrange(0, len(data), self.audioBytes)) * 100 + self.maxVolume // 2) / self.maxVolume
                # print "%.2f" % volume
                self.lastSecondVolumes[chunkInSecond] = volume # Logging the sound volume during the last second
                chunkInSecond = (chunkInSecond + 1) % self.chunksInSecond

                if volume >= self.volumeTreshold: # The chunk is loud enough
                    if not self.recording: # Start recording
                        # ToDo: check inputStream.get_time(), latency etc. to provide exact time stamp for file naming
                        self.fileName = strftime(self.fileNameFormat)
                        self.logger.info("%s recording started" % self.fileName)
                        self.recording = True
                        self.sample = ''
                        self.localMaxVolume = volume
                        self.audioFileLength = 0
                    elif volume > self.localMaxVolume:
                        self.localMaxVolume = volume
                    self.sampleLength = 0
                    chunksOfSilence = 0
                    self.sample += data
                elif self.recording: # Check for stop recording
                    self.sample += data
                    chunksOfSilence += 1
                    if not self.sampleLength and chunksOfSilence > self.chunksOfFadeout: # Enough silence for a trail
                        self.sampleLength = len(self.sample) # Removing extra silence at the end
                    if chunksOfSilence > self.chunksToStop: # Enough silence to stop recording
                        if self.quitAfterRecording:
                            self.inLoop = False
        except Exception, e:
            self.logger.warning("Processing error: %s: %s" % (e.__class__.__name__, e))
        except KeyboardInterrupt:
            self.logger.warning("Ctrl-C detected, exiting")
        self.inLoop = False
        self.logger.info("Done")

    def sigTerm(self):
        '''SIGTERM handler.'''
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

def usage(error = None):
    '''Prints usage information (preceded by optional error message) and exits with code 2.'''
    print "%s\n" % TITLE
    if error:
        print error
    print "Usage: python RCProcess.py [-c configFileName] [-h]"
    print "\t-c --config <filename>   Configuration file to use, defaults to %s" % DEFAULT_CONFIG_FILE_NAME
    print "\t-h --help                Show this help message"
    exit(2)

def main():
    RCProcess().run()

if __name__ == '__main__':
    main()
