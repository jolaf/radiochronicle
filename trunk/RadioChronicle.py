#
# RadioChronicle
#
# by Vladimir Yashunsky (vladimir.yashunsky@gmail.com)
# and Vasily Zakharov (vmzakhar@gmail.com)
# http://code.google.com/p/radiochronicle/
#
# Version 0.5
#
# Requires PyAudio: http://people.csail.mit.edu/hubert/pyaudio/
#

from ConfigParser import ConfigParser, NoOptionError, NoSectionError
from getopt import getopt, GetoptError
from logging import getLogger, StreamHandler, NOTSET
from logging.config import fileConfig
from signal import signal, SIGTERM
from struct import unpack
from sys import argv, exit
from thread import start_new_thread
from time import sleep, strftime
from traceback import format_exc
from os import remove
import wave

TITLE = "RadioChronicle v0.3  http://code.google.com/p/radiochronicle"
DEFAULT_CONFIG_FILE_NAME = 'rc.conf'

MONO = -1
STEREO = 0
LEFT = 1
RIGHT = 2

CHANNEL_NUMBERS = { 'LEFT': LEFT, 'RIGHT': RIGHT, 'STEREO': STEREO, 'ALL': STEREO, 'MONO': MONO }
CHANNEL_NAMES = { LEFT: 'LEFT', RIGHT: 'RIGHT', STEREO: 'STEREO', MONO: 'MONO' }

PACK_FORMATS = { 8 : 'b', 16 : '<h', 32 : '<i' }

def mean(iterable):
    '''Returns arithmetic mean of numbers in the specified iterable.'''
    if not iterable:
        return 0
    sum = 0
    for (n, i) in enumerate(iterable, 1):
        sum += i
    return float(sum) / n

class RadioChronicle:
    # Default parameter values
    fileNameFormat = './RC-%Y%m%d-%H%M%S.wav'
    monitor = False
    volumeTreshold = 10
    maxPauseLength = 2
    trailLength = 0.1
    minRecordingLength = 0
    chunkSize = 1024
    inputDevice = None
    outputDevice = None
    audioBits = 16
    sampleRate = 44100
    inputStream = None
    outputStream = None
    audio = None
    logger = None

    def __init__(self):
        '''Fully constructs class instance, including reading configuration file and configuring audio devices.'''
        try: # Reading command line options
            configFileName = DEFAULT_CONFIG_FILE_NAME
            (options, args) = getopt(argv[1:], 'c:h', ('config=', 'help'))
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
            channel = 'MONO'
            try:
                section = 'general'
                try:
                    self.fileNameFormat = config.get(section, 'fileNameFormat').strip()
                except NoOptionError: pass
                try:
                    self.monitor = config.getboolean(section, 'monitor')
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].monitor: '%s', must be 1/yes/true/on or 0/no/false/off" % (section, config.get(section, 'monitor')))
            except NoSectionError: pass
            try:
                section = 'tuning'
                try:
                    value = config.get(section, 'volumeTreshold')
                    self.volumeTreshold = float(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].volumeTreshold: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'maxPauseLength')
                    self.maxPauseLength = float(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].maxPauseLength: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'minRecordingLength')
                    self.minRecordingLength = float(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].minRecordingLength: '%s', must be a float" % (section, value))
                try:
                    value = config.get(section, 'trailLength')
                    self.trailLength = float(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].trailLength: '%s', must be a float" % (section, value))
            except NoSectionError: pass
            try:
                section = 'device'
                try:
                    value = config.get(section, 'chunkSize')
                    self.chunkSize = int(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].chunkSize: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'inputDevice')
                    self.inputDevice = int(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].inputDevice: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'outputDevice')
                    self.outputDevice = int(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].outputDevice: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'audioBits')
                    self.audioBits = int(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].audioBits: '%s', must be an integer" % (section, value))
                try:
                    value = config.get(section, 'sampleRate')
                    self.sampleRate = int(value)
                except NoOptionError: pass
                except ValueError, e:
                    raise ValueError("Bad value for [%s].sampleRate: '%s', must be an integer" % (section, value))
                try:
                    channel = config.get(section, 'channel') # Will be processed later
                except NoOptionError: pass
            except NoSectionError: pass

            # Validating configuration parameters
            if not self.fileNameFormat:
                raise ValueError("Bad value for fileNameFormat: must be not empty")
            if not 0 <= self.volumeTreshold <= 100:
                raise ValueError("Bad value for volumeTreshold: %.2f, must be 0-100" % self.volumeTreshold)
            if self.maxPauseLength < 0:
                self.maxPauseLength = 0
            if self.minRecordingLength <0:
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

    def __del__(self):
        '''Frees the PyAudio resources.'''
        if self.audio:
            self.closeInputStream()
            self.closeOutputStream()
            self.audio.terminate()
            self.logger.debug("destroyed")

    def deviceInfo(self, device = None, expectOutput = None):
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
        if expectOutput != None and isOutput != expectOutput:
            raise ValueError
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
            # Note: After 5-10 occurences of the above exception system hangs, so stream re-create seems necessary
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

    def saveSample(self):
        '''Saves the curent sample to the audio file.
           If the file does not exists, it is created.
           If the sample length is not equal to the self.sampleLength value, it means, we've cut
           the silence at the end of the sample, so it's the end of the file and it should be closed.
           The function returns True on success or if the recording is off, False otherwise.'''
        if not self.recording:
            return True
        try:
            if not self.audioFile: # Creating the file if necessary
                self.audioFile = wave.open(self.fileName, 'wb')
                self.audioFile.setnchannels(self.numOutputChannels)
                self.audioFile.setsampwidth(self.audioBytes)
                self.audioFile.setframerate(self.sampleRate)

            if self.sampleLength:
                finalSample = True
            else:
                # If sampleLength wasn't set manualy, all the sample is saved.
                # It means the recording isn't over yet.
                self.sampleLength = len(self.sample)
                finalSample = False

            self.audioFile.writeframes(self.sample[:self.sampleLength]) # Removing extra silence at the end, if needed

            self.sample = ''
            self.sampleLength = 0

            if finalSample:
                self.recording = False
                self.audioFile.close()
                self.audioFile = None
                recordLength = (float(self.audioFileLength) / self.outputSecondSize)-self.maxPauseLength+self.trailLength
                if recordLength >= self.minRecordingLength:
                    self.logger.info("Recording finished, max volume %.2f, %.1f seconds" % (self.localMaxVolume, recordLength))
                else:
                    try:
                        remove(self.fileName)
                        self.logger.info("The record is deleted for beeing too short (%.1f seconds)" % (recordLength))
                    except Exception, e:
                        self.logger.warning("Error deleting file: %s: %s" % (e.__class__.__name__, e))
                        return False

            return True
        except Exception, e:
            self.logger.warning("File output error: %s: %s" % (e.__class__.__name__, e))
            return False

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
                volume = (mean(abs(unpack(self.packFormat, data[i : i + self.audioBytes])[0]) for i in xrange(0, len(data), self.audioBytes)) * 100 + self.maxVolume / 2) / self.maxVolume
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
                    self.saveSample()
                    self.audioFileLength += len(data)
                elif self.recording: # Check for stop recording
                    self.sample += data
                    self.audioFileLength += len(data)
                    chunksOfSilence += 1
                    if not self.sampleLength and chunksOfSilence > self.chunksOfFadeout: # Enough silence for a trail
                        self.sampleLength = len(self.sample) # Removing extra silence at the end
                    if chunksOfSilence > self.chunksToStop: # Enough silence to stop recording
                        self.saveSample() # Stopping recording
                        if self.quitAfterRecording:
                            self.inLoop = False
        except Exception, e:
            self.logger.warning("Processing error: %s: %s" % (e.__class__.__name__, e))
        except KeyboardInterrupt, e:
            self.logger.warning("Ctrl-C detected at input, exiting")
        self.inLoop = False
        self.saveSample()   #This function call is executed on exit. So we can not be sure about the final sample volume params
        if self.audioFile:  #that's wy we'd better close the file manualy
            self.audioFile.close()
            self.audioFile = None
        self.closeInputStream()
        self.closeOutputStream()
        self.logger.info("Done")

    def commandConsole(self):
        '''Runs in a separate thread to provide a command line operation adjustments.'''
        try:
            while self.inLoop:
                inp = raw_input().split(' ')
                command = inp[0].lower()
                if 'help'.startswith(command):
                    print """\nAvailable console commands (first letter is enough):
Help               - Show this information
Exit/Quit          - Exit the program immediately
Last               - Exit the program after completion of the current file
Volume             - Print the current mean volume level
Monitor [on/off]   - Show or toggle monitor status
Threshold [value]  - Show or set the volume threshold level\n"""
                elif 'exit'.startswith(command) or 'quit'.startswith(command):
                    self.logger.info("Exiting")
                    self.inLoop = False
                elif 'volume'.startswith(command):
                    print "%.2f%%" % mean(self.lastSecondVolumes) # Using print for non-functional logging
                elif 'monitor'.startswith(command):
                    if len(inp) < 2:
                        print "Monitor is %s" % ('ON' if self.monitor else 'OFF') # Using print for non-functional logging
                    else:
                        self.monitor = inp[1].lower().strip() in ('true', 'yes', 'on', '1')
                        self.logger.info("Monitor is set to %s" % ('ON' if self.monitor else 'OFF'))
                elif 'last'.startswith(command):
                    if self.recording:
                        self.quitAfterRecording = True
                        self.logger.info("Going to exit after the end of the recording")
                    else:
                        self.logger.info("Exiting")
                        self.inLoop = False
                elif 'threshold'.startswith(command):
                    if len(inp) < 2:
                        print "Current volume treshold: %.2f" % self.volumeTreshold # Using print for non-functional logging
                    else:
                        try:
                            self.volumeTreshold = float(inp[1])
                            if not 0 <= self.volumeTreshold <= 100:
                                raise
                            self.logger.info("New volume treshold: %.2f" % self.volumeTreshold)
                        except:
                            print "Bad value, expected 0-100" # Using print for non-functional logging
        except EOFError, e:
            self.logger.warning("Console EOF detected")
        except Exception, e:
            self.logger.warning("Console error: %s: %s\n%s" % (e.__class__.__name__, e, format_exc()))
            self.inLoop = False
        except KeyboardInterrupt, e:
            self.logger.warning("Ctrl-C detected at console, exiting")
            self.inLoop = False

    def sigTerm(self):
        '''SIGTERM handler.'''
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

def usage(error = None):
    '''Prints usage information (preceded by optional error message) and exits with code 2.'''
    print "%s\n" % TITLE
    if error:
        print error
    print "Usage: python RadioChronicle.py [-c configFileName] [-h]"
    print "\t-c --config <filename>   Configuration file to use, defaults to %s" % DEFAULT_CONFIG_FILE_NAME
    print "\t-h --help                Show this help message"
    exit(2)

def main():
    RadioChronicle().run()

if __name__ == '__main__':
    main()
