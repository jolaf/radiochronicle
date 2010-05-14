#
# RadioChronicle
#
# Version 0.3
#
# Requires PyAudio: http://people.csail.mit.edu/hubert/pyaudio/
#

from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from getopt import getopt, GetoptError
from logging import getLogger, StreamHandler
from logging.config import fileConfig
from signal import signal, SIGTERM
from struct import unpack
from sys import argv, exit
from thread import start_new_thread
from time import sleep, strftime
from traceback import format_exc
import wave

TITLE = "RadioChronicle v0.3  http://code.google.com/p/radiochronicle"
DEFAULT_CONFIG_FILE_NAME = 'rc.conf'

MONO = -1
STEREO = 0
LEFT = 1
RIGHT = 2

CHANNELS_MAP = { 'LEFT': LEFT, 'RIGHT': RIGHT, 'STEREO': STEREO, 'ALL': STEREO, 'MONO': MONO }

PACK_FORMATS = { 8 : 'b', 16 : '<h', 32 : '<i' }

def mean(iterable):
    '''Returns integer arithmetic mean of numbers in the specified iterable.'''
    if not iterable:
        return 0
    sum = 0
    for (n, i) in enumerate(iterable, 1):
        sum += i
    return int(round(float(sum) / n))

def integral(iterable):
    '''Returns integral value of numbers in the specified iterable.
       Planned to be better than mean() above.'''
    # ToDo: currently it's not working, needs to be fixed
    # When it's ok, it's gonna replace the mean()
    return reduce(lambda y, x: 0.95 * y + 0.009 * x, iterable, 0)

class RadioChronicle:
    # Default parameter values
    fileNameFormat = './RC-%Y%m%d-%H%M%S.wav'
    monitor = False
    volumeTreshold = 10
    maxPauseLength = 2
    trailLength = 0.1
    chunkSize = 1024
    inputDevice = None
    outputDevice = None
    audioBits = 16
    sampleRate = 44100
    inputStream = None
    outputStream = None

    def __init__(self):
        '''Fully constructs class instance, including reading configuration file and configuring audio devices.'''
        # Reading config file and configuring logging
        try:
            configFileName = DEFAULT_CONFIG_FILE_NAME
            try:
                (options, args) = getopt(argv[1:], 'c:vh', ['config=', 'help'])
            except GetoptError, e:
                self.usage(e)
            for (option, value) in options:
                if option in ('-c', '--config'):
                    configFileName = value.strip()
                else:
                    self.usage()
            config = ConfigParser()
            # ToDo: What if config file is absent? Use config.readfp? Print warning?
            config.read(configFileName) # Doesn't produce errors if file is missing or damaged
            if config.has_section('loggers'):
                fileConfig(configFileName)
            self.logger = getLogger()
            if not self.logger.handlers:
                # ToDo: This doesn't work - if config file is absent, logging works poorly, needs fixing
                self.logger.addHandler(StreamHandler())
            signal(SIGTERM, self.sigTerm)
        except Exception, e:
            print "%s\n\nConfiguration error: %s" % (TITLE, e)
            exit(1)
        # Above this point, use print for diagnostics
        # From this point on, we have self.logger to use instead
        self.logger.info(TITLE)
        self.logger.info("Using %s", configFileName)
        print # Empty line to console only
        try:
            # Applying configuration
            channel = MONO
            # ToDo: need to catch ValueError(s) and provide information about which parameter was wrong
            try:
                section = 'general'
                try:
                    self.fileNameFormat = config.get(section, 'fileNameFormat')
                except NoOptionError: pass
                try:
                    self.monitor = config.getboolean(section, 'monitor')
                except NoOptionError: pass
            except NoSectionError: pass
            try:
                section = 'tuning'
                try:
                    self.volumeTreshold = config.getint(section, 'volumeTreshold')
                except NoOptionError: pass
                try:
                    self.maxPauseLength = config.getfloat(section, 'maxPauseLength')
                except NoOptionError: pass
                try:
                    self.trailLength = config.getfloat(section, 'trailLength')
                except NoOptionError: pass
            except NoSectionError: pass
            try:
                section = 'device'
                try:
                    self.chunkSize = config.getint(section, 'chunkSize')
                except NoOptionError: pass
                try:
                    self.inputDevice = config.getint(section, 'inputDevice')
                except NoOptionError: pass
                try:
                    self.outputDevice = config.getint(section, 'outputDevice')
                except NoOptionError: pass
                try:
                    self.audioBits = config.getint(section, 'audioBits')
                except NoOptionError: pass
                try:
                    self.sampleRate = config.getint(section, 'sampleRate')
                except NoOptionError: pass
                try:
                    channel = config.get(section, 'channel') # Will be processed later
                except NoOptionError: pass
            except NoSectionError: pass

            # Validating configuration parameters
            if not self.fileNameFormat:
                raise ValueError("File name format is empty")
            if not 0 <= self.volumeTreshold <= 100:
                raise ValueError("volumeTreshold must be 0-100, found %d" % self.volumeTreshold)
            if self.maxPauseLength < 0:
                self.maxPauseLength = 0
            if self.trailLength < 0:
                self.trailLength = 0
            if self.chunkSize < 1:
                raise ValueError("chunkSize must be 1 or more, found %d" % self.chunkSize)
            if self.inputDevice < 0:
                self.inputDevice = None
            if self.outputDevice < 0:
                self.outputDevice = None
            if self.audioBits not in (8, 16, 32):
                raise ValueError("audioBits must be 8, 16, or 32, found %d" % self.audioBits)
            if self.sampleRate < 1:
                raise ValueError("chunkSize must be positive, found %d" % self.sampleRate)
            try:
                self.channel = int(channel)
                if self.channel <= 0:
                    self.channel = None # Exception will be thrown below
            except ValueError:
                self.channel = CHANNELS_MAP.get(channel.upper().strip())
            if self.channel == None:
                raise ValueError("channel must be LEFT/RIGHT/STEREO/ALL/MONO or a number of 1 or more, found %s" % channel)

            # Accessing PyAudio engine
            try:
                from pyaudio import PyAudio
            except ImportError, e:
                raise ImportError("%s: %s\nPlease install PyAudio: http://people.csail.mit.edu/hubert/pyaudio" % (e.__class__.__name__, e))
            self.audio = PyAudio()
            self.logger.info(self.deviceInfo())
            print # Empty line to console only

            # Calculating derivative paratemers
            self.numInputChannels = 1 if self.channel == MONO else self.audio.get_device_info_by_index(self.inputDevice)['maxInputChannels']
            assert self.numInputChannels > 0
            self.numOutputChannels = self.numInputChannels if self.channel == STEREO else 1
            assert self.numOutputChannels > 0

            self.audioBytes = self.audioBits / 8
            self.maxVolume = 1 << (self.audioBits - 1)
            self.audioFormat = self.audio.get_format_from_width(self.audioBytes, False)
            self.packFormat = PACK_FORMATS[self.audioBits]

            self.inputBlockSize = self.numInputChannels * self.chunkSize * self.audioBytes
            self.outputBlockSize = self.numOutputChannels * self.chunkSize * self.audioBytes
            self.inputSecondSize = self.numInputChannels * self.sampleRate * self.audioBytes
            self.outputSecondSize = self.numOutputChannels * self.sampleRate * self.audioBytes
            self.chunksInSecond = self.sampleRate / self.chunkSize
            self.chunksToStop = self.chunksInSecond * self.maxPauseLength
            self.chunksOfFadeout = self.chunksInSecond * self.trailLength

            # Printing info on audio devices to use
            if self.inputDevice != None:
                self.logger.info("Using input device %s" % self.deviceInfo(self.inputDevice))
            else:
                self.logger.info("Using default input device %s" % self.deviceInfo(self.audio.get_default_input_device_info()))
            if self.outputDevice != None:
                self.logger.info("Using output device %s" % self.deviceInfo(self.outputDevice))
            else:
                self.logger.info("Using default output device %s" % self.deviceInfo(self.audio.get_default_output_device_info()))

            # Diagnosting audio devices
            if not self.createInputStream():
                raise Exception("Can't create input stream, exiting")
            self.closeInputStream()
            if not self.createOutputStream():
                raise Exception("Can't create output stream, exiting")
            self.closeOutputStream()

            # ToDo: Print actual configuration parameter values
        except Exception, e:
            self.logger.error("Configuration error: %s" % e)
            exit(1)

    def __del__(self):
        '''Frees the PyAudio resources.'''
        self.closeInputStream()
        self.closeOutputStream()
        self.audio.terminate()
        self.logger.debug("Destroyed")

    def deviceInfo(self, device = None):
        '''Provides string information about system audio device(s).'''
        if device == None:
            # Return info on all available devices
            inputDevices = []
            outputDevices = []
            for i in xrange(self.audio.get_device_count()):
                device = self.audio.get_device_info_by_index(i)
                if device['maxOutputChannels']:
                    outputDevices.append(device)
                else:
                    inputDevices.append(device)
            return "Detected audio input devices:\n%s\nDetected audio output devices:\n%s" % ('\n'.join(self.deviceInfo(device) for device in inputDevices), '\n'.join(self.deviceInfo(device) for device in outputDevices))
        # else Return info on a particular device
        if type(device) == int:
            device = self.audio.get_device_info_by_index(device)
        inputChannels = device['maxInputChannels']
        outputChannels = device['maxOutputChannels']
        isOutput = bool(outputChannels)
        assert bool(inputChannels) != isOutput
        return "%d: %s (%d channels)" % (device['index'], device['name'], outputChannels if isOutput else inputChannels)

    def createInputStream(self):
        '''Creates an input stream if it doesn't already exist.
           Returns True if stream already exists or was created successfuly, False otherwise.'''
        if self.inputStream:
            return True
        try:
            self.inputStream = self.audio.open(self.sampleRate, self.numInputChannels, self.audioFormat, True, False, self.inputDevice, None, self.chunkSize)
            return True
        except Exception, e:
            self.logger.warning("Error creating input stream: %s: %s" % (e.__class__.__name__, e))
            return False

    def createOutputStream(self):
        '''Creates an output stream if it doesn't already exist.
           Returns True if stream already exists or was created successfuly, False otherwise.'''
        if self.outputStream:
            return True
        try:
            self.outputStream = self.audio.open(self.sampleRate, self.numOutputChannels, self.audioFormat, False, True, None, self.outputDevice, self.chunkSize)
            return True
        except Exception, e:
            self.logger.warning("Error creating output stream: %s: %s" % (e.__class__.__name__, e))
            return False

    def closeInputStream(self):
        if self.inputStream:
            self.inputStream.close()
            self.inputStream = None

    def closeOutputStream(self):
        if self.outputStream:
            self.outputStream.close()
            self.outputStream = None

    def readAudioData(self):
        '''Reads a chunk of audio data from the input stream.
           Returns the retrieved data if successful, None otherwise.'''
        if not self.createInputStream():
            return None
        try:
            data = self.inputStream.read(self.chunkSize)
            return data
        except Exception, e:
            # Note: IOError: [Errno Input overflowed] -9981 often occurs when running under debugger
            # Note: IOError: [Errno Unanticipated host error] -9999 occurs when audio device is removed (cable unplugged)
            # ToDo: After 5-10 occurences of the above exception system hangs, so stream re-create seems necessary
            self.logger.warning("Audio input error: %s: %s" % (e.__class__.__name__, e))
            self.closeInputStream()
            self.dump()
            return None

    def writeAudioData(self, data):
        '''Writes a chunk of audio data to the output stream.
           Returns True if successful, False otherwise.'''
        if not self.createOutputStream():
            return False
        try:
            self.outputStream.write(data)
            return True
        except Exception, e:
            self.logger.warning("Audio output error: %s: %s" % (e.__class__.__name__, e))
            self.closeOutputStream()
            return False

    def dump(self):
        '''If recording is on, dumps the recorded data (if any) to a .wav file, and stops recording.
           Returns True if dump was successful or recording was off, False otherwise.'''
        if not self.recording:
            return True
        try:
            self.recording = False
            if not self.sampleLength:
                self.sampleLength = len(self.sample)
            f = wave.open(self.fileName, 'wb')
            f.setnchannels(self.numOutputChannels)
            f.setsampwidth(self.audioBytes)
            f.setframerate(self.sampleRate)
            f.writeframes(self.sample[:self.sampleLength]) # Removing extra silence at the end
            f.close()
            self.logger.info("Recording finished, max volume %d, %.1f seconds" % (self.localMaxVolume, float(self.sampleLength) / self.outputSecondSize))
            return True
        except:
            self.logger.warning("File output error: %s: %s" % (e.__class__.__name__, e))
            return False

    def run(self):
        '''Runs main audio processing loop.'''
        self.inLoop = True
        self.recording = False
        self.quitAfterRecording = False
        self.lastSecondVolumes = [0] * self.chunksInSecond
        chunkInSecond = 0
        start_new_thread(self.commandConsole, ()) # Start command console thread
        print # Empty line to console only
        self.logger.info("Listening started")

        # Main audio processing loop
        while self.inLoop:
            try:
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
                volume = (mean(abs(unpack(self.packFormat, data[i : i + self.audioBytes])[0]) for i in xrange(0, len(data), self.audioBytes)) * 100 + self.maxVolume / 2) / self.maxVolume
                self.lastSecondVolumes[chunkInSecond] = volume # Logging the sound volume during the last second
                chunkInSecond = (chunkInSecond + 1) % self.chunksInSecond

                if volume >= self.volumeTreshold: # The chunk is loud enough
                    if not self.recording: # Start recording
                        # ToDo: add mechanism to avoid recording of very short transmissions
                        # ToDo: check inputStream.get_time(), latency etc. to provide exact time stamp for file naming
                        # ToDo: check actual recording delay, is it really essential?
                        self.fileName = strftime(self.fileNameFormat)
                        self.logger.info("%s recording started" % self.fileName)
                        self.recording = True
                        self.sample = ''
                        self.localMaxVolume = volume
                    elif volume > self.localMaxVolume:
                        self.localMaxVolume = volume
                    self.sample += data
                    chunksOfSilence = 0
                    self.sampleLength = 0
                elif self.recording: # Check for stop recording
                    self.sample += data
                    chunksOfSilence += 1
                    if not self.sampleLength and chunksOfSilence > self.chunksOfFadeout: # Enough silence for a trail
                        self.sampleLength = len(self.sample) # Removing extra silence at the end
                    if chunksOfSilence > self.chunksToStop: # Enough silence to stop recording
                        self.dump() # Stopping recording
                        if self.quitAfterRecording:
                            self.inLoop = False
            except Exception, e:
                self.logger.warning("Processing error: %s: %s" % (e.__class__.__name__, e))
                self.dump()
            except KeyboardInterrupt, e:
                self.logger.warning("Ctrl-C detected at input, exiting")
                self.inLoop = False

        if self.recording: # Listening was interrupted in the middle of recording
            self.dump()
        self.closeInputStream()
        self.closeOutputStream()
        self.logger.info("Done")

    def commandConsole(self):
        '''Runs in a separate thread to provide a command line operation adjustments.'''
        while self.inLoop:
            try:
                inp = raw_input().split(' ')
                command = inp[0]
                if inp[0] == 'mean':
                    print "%d%%" % mean(self.lastSecondVolumes) # Using print for non-functional logging
                elif inp[0] in ('exit', 'quit'):
                    self.logger.info("Exiting")
                    self.inLoop = False
                elif inp[0] == 'monitor':
                    if len(inp) < 2:
                        print "Monitor is %s" % ('ON' if self.monitor else 'OFF') # Using print for non-functional logging
                    else:
                        self.monitor = inp[1].lower().strip() in ('true', 'on', '1')
                        self.logger.info("Monitor is set to %s" % ('ON' if self.monitor else 'OFF'))
                elif inp[0] == 'last':
                    if self.recording:
                        self.quitAfterRecording = True
                        self.logger.info("Going to exit after the end of the recording")
                    else:
                        self.logger.info("Exiting")
                        self.inLoop = False
                elif inp[0] == 'threshold':
                    if len(inp) < 2:
                        print "Current volume treshold: %d" % self.volumeTreshold # Using print for non-functional logging
                    else:
                        try:
                            self.volumeTreshold = int(inp[1])
                            if not 0 <= self.volumeTreshold <= 100:
                                raise
                            self.logger.info("New volume treshold: %d" % self.volumeTreshold)
                        except:
                            print "Bad value, expected 0-100" # Using print for non-functional logging
            except EOFError, e:
                self.logger.warning("Console EOF detected, deactivating")
                break
            except Exception, e:
                self.logger.warning("Console error: %s: %s\n%s" % (e.__class__.__name__, e, format_exc()))
            except KeyboardInterrupt, e:
                self.logger.warning("Ctrl-C detected at console, exiting")
                self.inLoop = false

    def usage(self, error = None):
        '''Prints usage information (preceded by optional error message) and exits with code 2.'''
        if error:
            print error
        print "Usage: python RadioChronicle.py [-c configFileName] [-h]"
        print "\t-h --help  Show this help message"
        print "\t-c --config <filename> Configuration file to use, defaults to %s" % DEFAULT_CONFIG_FILE_NAME
        exit(2)

    def sigTerm(self):
        '''SIGTERM handler.'''
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

def main():
    RadioChronicle().run()

if __name__ == '__main__':
    main()
