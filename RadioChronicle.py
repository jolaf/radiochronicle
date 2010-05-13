#
# RadioChronicle
#
# Version 0.3
#
# Requires PyAudio: http://people.csail.mit.edu/hubert/pyaudio/
#

import logging
import logging.config
import sys
import wave

from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from getopt import getopt, GetoptError
from signal import signal, SIGTERM
from struct import unpack
from thread import start_new_thread
from time import sleep, strftime
from traceback import format_exc

TITLE = "RadioChronicle v0.3  http://code.google.com/p/radiochronicle"
DEFAULT_CONFIG_FILE_NAME = "rc.conf"

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
       Should be better than mean() above).'''
    # ToDo: currently it's not working, needs to be fixed
    # When it's ok, it's gonna replace the mean()
    return reduce(lambda y, x: 0.95 * y + 0.009 * x, iterable, 0)

class RadioChronicle:
    fileNameFormat = "RC-%Y%m%d-%H%M%S.wav"
    monitor = False
    volumeTreshold = 10
    maxPauseLength = 2
    trailLength = 0.1
    chunkSize = 1024
    inputDevice = None
    outputDevice = None
    audioBits = 16
    sampleRate = 44100

    def __init__(self):
        # Reading config file and configuring logging
        try:
            configFileName = DEFAULT_CONFIG_FILE_NAME
            try:
                (options, args) = getopt(sys.argv[1:], 'c:vh', ['config=', 'help'])
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
                logging.config.fileConfig(configFileName)
            self.logger = logging.getLogger()
            if not self.logger.handlers:
                # ToDo: This doesn't work, if config file is absent, logging works poorly, needs fixing
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(handler)
            self.logger.info("%s\nUsing %s\n", TITLE, configFileName)
            signal(SIGTERM, self.sigTerm)
        except Exception, e:
            print "%s\n\nERROR: %s" % (TITLE, e)
            sys.exit(1)
        # From this point on, we have self.logger to use for diagnostics
        try:
            channel = MONO

            # Reading configuration
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

            # Validating parameters
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

            # Accessing Audio engine
            try:
                from pyaudio import PyAudio
            except ImportError, e:
                raise ImportError("%s: %s\nPlease install PyAudio: http://people.csail.mit.edu/hubert/pyaudio" % (e.__class__.__name__, e))
            self.audio = PyAudio()
            self.logger.info("%s\n" % self.deviceInfo())

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

            if self.inputDevice != None:
                self.logger.info("Using input device %s" % self.deviceInfo(self.inputDevice))
            else:
                self.logger.info("Using default input device %s" % self.deviceInfo(self.audio.get_default_input_device_info()))
            if self.outputDevice != None:
                self.logger.info("Using output device %s" % self.deviceInfo(self.outputDevice))
            else:
                self.logger.info("Using default output device %s" % self.deviceInfo(self.audio.get_default_output_device_info()))
        except Exception, e:
            self.logger.error("ERROR: %s" % e)
            sys.exit(1)

    def usage(self, error = None):
        if error:
            print error
        print "Usage: python RadioChronicle.py [-c configFileName] [-h]"
        print "\t-h --help  Show this help message"
        print "\t-c --config <filename> Configuration file to use, defaults to %s" % DEFAULT_CONFIG_FILE_NAME
        sys.exit(2)

    def sigTerm(self):
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

    def deviceInfo(self, device = None):
        if device == None:
            inputDevices = []
            outputDevices = []
            for i in xrange(self.audio.get_device_count()):
                device = self.audio.get_device_info_by_index(i)
                if device['maxOutputChannels']:
                    outputDevices.append(device)
                else:
                    inputDevices.append(device)
            return "Detected audio input devices:\n%s\nDetected audio output devices:\n%s" % ('\n'.join(self.deviceInfo(device) for device in inputDevices), '\n'.join(self.deviceInfo(device) for device in outputDevices))
        if type(device) == int:
            device = self.audio.get_device_info_by_index(device)
        inputChannels = device['maxInputChannels']
        outputChannels = device['maxOutputChannels']
        isOutput = bool(outputChannels)
        assert bool(inputChannels) != isOutput
        return "%d: %s (%d channels)" % (device['index'], device['name'], outputChannels if isOutput else inputChannels)

    def commandConsole(self):
        try:
            while self.inLoop:
                try:
                    inp = raw_input().split(' ')
                    command = inp[0]
                    if inp[0] == "mean":
                        print "%d%%" % mean(self.lastSecondVolumes)
                    elif inp[0] == "quit":
                        self.inLoop = False
                    elif inp[0] == "monitor":
                        if len(inp) < 2:
                            print "Monitor is %s" % 'ON' if self.monitor else 'OFF'
                        else:
                            self.monitor = inp[1].lower().strip() in ('true', 'on', '1')
                            print "Monitor is set to %s" % 'ON' if self.monitor else 'OFF'
                    elif inp[0] == "last":
                        if self.recording:
                            self.quitAfterRecording = True
                            print "Recording the last track"
                        else:
                            self.inLoop = False
                    elif inp[0] == "level":
                        if len(inp) < 2:
                            print "Current silence level: %d" % self.volumeTreshold
                        else:
                            try:
                                self.volumeTreshold = int(inp[1])
                                assert 0 <= self.volumeTreshold <= 100
                                print "New silence level: %d" % self.volumeTreshold
                            except:
                                print "Bad value, expected 0-100"
                except Exception, e:
                    self.logger.warning("Console error: %s: %s\n%s" % (e.__class__.__name__, e, format_exc()))
        except KeyboardInterrupt, e:
            self.logger.warning("Ctrl-C detected at console, exiting")
            self.inLoop = false

    def dump(self):
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
        except:
            self.logger.warning("Output error: %s: %s\n%s" % (e.__class__.__name__, e, format_exc()))

    def run(self):
        self.inLoop = True
        self.recording = False
        self.quitAfterRecording = False
        self.lastSecondVolumes = [0] * self.chunksInSecond
        frameInSecond = 0

        self.logger.info("\nListening started")
        inputStream = self.audio.open(self.sampleRate, self.numInputChannels, self.audioFormat, True, False, self.inputDevice, None, self.chunkSize)
        outputStream = self.audio.open(self.sampleRate, self.numOutputChannels, self.audioFormat, False, True, None, self.outputDevice, self.chunkSize)

        #start_new_thread(self.commandConsole, ())
        try:
            while self.inLoop:
                # We've just received a new part of sound data
                try:
                    data = inputStream.read(self.chunkSize)
                    # ToDo: check inputStream.get_time(), latency etc. to provide exact time stamp for file naming
                    assert len(data) == self.inputBlockSize

                    if self.channel not in (MONO, STEREO):
                        data = ''.join(data[i : i + self.audioBytes] for i in xrange((self.channel - 1) * self.audioBytes, len(data), self.numInputChannels * self.audioBytes))
                    assert len(data) == self.outputBlockSize

                    if self.monitor: # Provide monitor output
                        # ToDo: Need separate output stream to allow only left/right channel output
                        outputStream.write(data, self.chunkSize)

                    # Let's see if data contains something loud enough
                    volume = (mean(abs(unpack(self.packFormat, data[i : i + self.audioBytes])[0]) for i in xrange(0, len(data), self.audioBytes)) * 100 + self.maxVolume / 2) / self.maxVolume
                    self.lastSecondVolumes[frameInSecond] = volume # Logging the sound volume during the last second
                    frameInSecond = (frameInSecond + 1) % self.chunksInSecond

                    if volume >= self.volumeTreshold:
                        if not self.recording: # Start recording
                            # ToDo: add mechanism to avoid recording of very short transmissions
                            self.fileName = strftime(self.fileNameFormat)
                            print "%s recording started" % self.fileName
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
                                inLoop = False
                except Exception, e:
                    # Note: IOError: [Errno Input overflowed] -9981 often occurs when running under debugger
                    # Note: IOError: [Errno Unanticipated host error] -9999 occurs when audio device is removed (cable unplugged)
                    self.logger.warning("Input error: %s: %s" % (e.__class__.__name__, e))
                    # ToDo: After 5-10 exceptions system hangs, stream re-create may be necessary
                    # ToDo: Also, file being recorded must be flushed and closed immediately
        except KeyboardInterrupt, e:
            self.logger.warning("Ctrl-C detected at input, exiting")

        if self.recording: # Listening was interrupted in the middle of recording
            self.dump()
        inputStream.close()
        outputStream.close()
        self.audio.terminate()
        self.logger.info("Done")

def main():
    RadioChronicle().run()

if __name__ == '__main__':
    main()
