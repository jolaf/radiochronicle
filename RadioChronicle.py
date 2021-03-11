#!/usr/bin/python3
#
# RadioChronicle.py
#
# A part of RadioChronicle project
#
# by Vladimir Yashunsky (vladimir.yashunsky@gmail.com)
# and Vasily Zakharov (vmzakhar@gmail.com)
# https://github.com/jolaf/radiochronicle
#
# Version 0.6
#
# Requires PyAudio: http://people.csail.mit.edu/hubert/pyaudio/
#
from configparser import ConfigParser, NoOptionError, NoSectionError
from getopt import getopt
from logging import getLogger, Logger, StreamHandler, NOTSET
from logging.config import fileConfig
from signal import signal, SIGTERM
from statistics import mean
from struct import unpack
from sys import argv, exit as sysExit
from _thread import start_new_thread
from time import sleep, strftime
from traceback import format_exc
from types import FrameType
from typing import cast, Final, List, Mapping, NoReturn, Optional, Union
from wave import open as waveOpen, Wave_write as WaveWriter

try:
    from pyaudio import PyAudio, Stream as AudioStream
except ImportError as e:
    raise ImportError(f"{type(e).__name__}: {e}\nPlease install PyAudio: http://people.csail.mit.edu/hubert/pyaudio") from e

TITLE: Final[str] = "RadioChronicle v0.6  https://github.com/jolaf/radiochronicle"
DEFAULT_CONFIG_FILE_NAME: Final[str] = 'rc.conf'

MONO: Final[int] = -1
STEREO: Final[int] = 0
LEFT: Final[int] = 1
RIGHT: Final[int] = 2

CHANNEL_NUMBERS: Final[Mapping[str, int]] = { 'LEFT': LEFT, 'RIGHT': RIGHT, 'STEREO': STEREO, 'ALL': STEREO, 'MONO': MONO }
CHANNEL_NAMES: Final[Mapping[int, str]] = { LEFT: 'LEFT', RIGHT: 'RIGHT', STEREO: 'STEREO', MONO: 'MONO' }

PACK_FORMATS: Final[Mapping[int, str]] = { 8 : 'b', 16 : '<h', 32 : '<i' }

class RadioChronicle:
    # Default parameter values
    fileNameFormat = './RC-%Y%m%d-%H%M%S.wav'
    monitor = False
    volumeTreshold = 10.0
    maxPauseLength = 2.0
    trailLength = 0.1
    minRecordingLength = 0.0
    chunkSize = 1024
    inputDevice: Optional[int] = None
    outputDevice: Optional[int] = None
    audioBits = 16
    sampleRate = 44100
    inputStream: Optional[AudioStream] = None
    outputStream: Optional[AudioStream] = None

    audio: PyAudio
    logger: Logger

    audioFile: Optional[WaveWriter]
    sample: bytes
    sampleLength: int
    audioFileLength: int
    inLoop: bool
    recording: bool
    quitAfterRecording: bool
    lastSecondVolumes: List[float]

    fileName: str
    localMaxVolume: float

    def __init__(self) -> None: # pylint: disable=too-complex, too-many-statements
        '''Fully constructs class instance, including reading configuration file and configuring audio devices.'''
        try: # Reading command line options
            configFileName = DEFAULT_CONFIG_FILE_NAME
            (options, _args) = getopt(argv[1:], 'c:h', ['config=', 'help'])
            for (option, optionValue) in options:
                if option in ('-c', '--config'):
                    configFileName = optionValue.strip()
                else:
                    usage()
        except Exception as e:
            usage(e)
        try: # Reading config file and configuring logging
            config = ConfigParser(interpolation = None, inline_comment_prefixes = (';',))
            config.read_file(open(configFileName)) # Using read_file(open()) to make sure file exists
            if config.has_section('loggers'):
                fileConfig(config)
            self.logger = getLogger()
            if not self.logger.handlers: # Provide default logger
                self.logger.addHandler(StreamHandler())
                self.logger.setLevel(NOTSET)
            signal(SIGTERM, self.sigTerm)
        except Exception as e:
            print(f"{TITLE}\n\nConfig error: {e}")
            print(format_exc())
            sysExit(1)
        # Above this point, use print for diagnostics
        # From this point on, we have self.logger to use instead
        self.logger.info(TITLE)
        self.logger.info(f"Using {configFileName}")
        print() # Empty line to console only
        try: # Applying configuration
            channel = 'MONO'
            value: str
            try:
                section = 'general'
                try:
                    self.fileNameFormat = config.get(section, 'fileNameFormat').strip()
                except NoOptionError:
                    pass
                try:
                    self.monitor = config.getboolean(section, 'monitor')
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].monitor: '{config.get(section, 'monitor')}', must be 1/yes/true/on or 0/no/false/off") from e
            except NoSectionError:
                pass
            try:
                section = 'tuning'
                try:
                    value = config.get(section, 'volumeTreshold')
                    self.volumeTreshold = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].volumeTreshold: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'maxPauseLength')
                    self.maxPauseLength = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].maxPauseLength: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'minRecordingLength')
                    self.minRecordingLength = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].minRecordingLength: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'trailLength')
                    self.trailLength = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].trailLength: '{value}', must be a float") from e
            except NoSectionError:
                pass
            try:
                section = 'device'
                try:
                    value = config.get(section, 'chunkSize')
                    self.chunkSize = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].chunkSize: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'inputDevice')
                    self.inputDevice = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].inputDevice: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'outputDevice')
                    self.outputDevice = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].outputDevice: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'audioBits')
                    self.audioBits = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].audioBits: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'sampleRate')
                    self.sampleRate = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].sampleRate: '{value}', must be an integer") from e
                try:
                    channel = config.get(section, 'channel') # pylint: disable=redefined-variable-type # Will be processed later
                except NoOptionError:
                    pass
            except NoSectionError:
                pass

            # Validating configuration parameters
            if not self.fileNameFormat:
                raise ValueError("Bad value for fileNameFormat: must be not empty")
            if not 0 <= self.volumeTreshold <= 100:
                raise ValueError(f"Bad value for volumeTreshold: {self.volumeTreshold:.2f}, must be 0-100")
            if self.maxPauseLength < 0:
                self.maxPauseLength = 0.0
            if self.minRecordingLength < 0:
                self.minRecordingLength = 0.0
            if self.trailLength < 0:
                self.trailLength = 0.0
            if self.chunkSize < 1:
                raise ValueError(f"Bad value for chunkSize: {self.chunkSize}, must be 1 or more")
            if self.inputDevice:
                if self.inputDevice == -1:
                    self.inputDevice = None
                elif self.inputDevice < -1:
                    raise ValueError(f"Bad value for input device: {self.inputDevice}, must be -1 or more")
            if self.outputDevice:
                if self.outputDevice == -1:
                    self.outputDevice = None
                elif self.outputDevice < -1:
                    raise ValueError(f"Bad value for output device: {self.outputDevice}, must be -1 or more")
            if self.audioBits not in (8, 16, 32):
                raise ValueError(f"Bad value for audioBits: {self.audioBits}, must be 8, 16, or 32")
            if self.sampleRate < 1:
                raise ValueError(f"Bad value for chunkSize: {self.sampleRate}, must be positive")
            try:
                intChannel: Optional[int] = int(channel)
                assert intChannel is not None
                if intChannel <= 0:
                    intChannel = None # Exception will be thrown below
            except ValueError:
                intChannel = CHANNEL_NUMBERS.get(channel.strip().upper()) # Would be None if not found
            if intChannel is None:
                raise ValueError(f"Bad value for channel: {channel}, must be LEFT/RIGHT/STEREO/ALL/MONO or a number of 1 or more")
            self.channel = intChannel

            # Accessing PyAudio engine
            self.audio = PyAudio()
            print(f"{self.deviceInfo()}\n") # Using print for non-functional logging

            # Accessing audio devices
            try:
                if self.inputDevice is not None:
                    inputDeviceInfo = self.audio.get_device_info_by_index(self.inputDevice)
                    self.logger.info(f"Using input device {self.deviceInfo(inputDeviceInfo, False)}")
                else:
                    inputDeviceInfo = self.audio.get_default_input_device_info()
                    self.logger.info(f"Using default input device {self.deviceInfo(inputDeviceInfo, False)}")
            except ValueError as e:
                raise ValueError(f"{f'Input device {self.inputDevice}' if self.inputDevice is not None else 'Default input device'} is not in fact an input device") from e
            except IOError as e:
                raise IOError(f"Can't access {f'input device {self.inputDevice}' if self.inputDevice is not None else 'default input device'}: {e}") from e
            try:
                if self.outputDevice is not None:
                    outputDeviceInfo = self.audio.get_device_info_by_index(self.outputDevice)
                    self.logger.info(f"Using output device {self.deviceInfo(outputDeviceInfo, True)}")
                else:
                    outputDeviceInfo = self.audio.get_default_output_device_info()
                    self.logger.info(f"Using default output device {self.deviceInfo(outputDeviceInfo, True)}")
            except ValueError as e:
                raise ValueError(f"{f'output device {self.outputDevice}' if self.outputDevice is not None else 'Default output device'} is not in fact an output device") from e
            except IOError as e:
                raise IOError(f"Can't access {f'output device {self.outputDevice}' if self.outputDevice is not None else 'default output device'}: {e}") from e
            print() # Empty line to console only

            # Calculating derivative paratemers
            self.numInputChannels = 1 if self.channel == MONO else cast(int, inputDeviceInfo['maxInputChannels'])
            assert self.numInputChannels > 0
            if self.channel > self.numInputChannels:
                raise ValueError(f"Bad value for channel: {self.channel}, must be no more than {self.numInputChannels}")
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
            self.logger.info(f"Recording {self.sampleRate}Hz/{self.audioBits}-bit/{CHANNEL_NAMES.get(self.channel) or f'channel {self.channel}'} to {self.fileNameFormat}")
            self.logger.info(f"Volume threshold {self.volumeTreshold:.2f}%, max pause {self.maxPauseLength:.1f} seconds, min recording length {self.minRecordingLength:.1f} seconds, trail {self.trailLength:.1f} seconds")
            self.logger.info(f"Monitor is {'ON' if self.monitor else 'OFF'}")
            print("Type 'help' for console commands reference") # Using print for non-functional logging
            print() # Empty line to console only
        except Exception as e:
            self.logger.error(f"Configuration error: {e}")
            print(format_exc())
            sysExit(1)

    def __del__(self) -> None:
        '''Frees the PyAudio resources.'''
        if self.audio:
            self.closeInputStream()
            self.closeOutputStream()
            self.audio.terminate()
            self.logger.debug("destroyed")

    def deviceInfo(self, device: Union[int, Mapping[str, Union[str, int, float]], None] = None, expectOutput: Optional[bool] = None) -> str:
        '''Provides string information about system audio device(s).'''
        if device is None:
            # Return info on all available devices
            inputDevices = []
            outputDevices = []
            for i in range(self.audio.get_device_count()):
                device = self.audio.get_device_info_by_index(i)
                if device['maxOutputChannels']:
                    outputDevices.append(device)
                if device['maxInputChannels']:
                    inputDevices.append(device)
            return '\n'.join(("Detected audio input devices:", '\n'.join(self.deviceInfo(device) for device in inputDevices), "\nDetected audio output devices:", '\n'.join(self.deviceInfo(device) for device in outputDevices)))
        # else Return info on a particular device
        if isinstance(device, int):
            device = self.audio.get_device_info_by_index(device)
        inputChannels = device['maxInputChannels']
        outputChannels = device['maxOutputChannels']
        if expectOutput is not None and not bool(outputChannels if expectOutput else inputChannels):
            raise ValueError
        return f"{device['index']}: {device['name']} ({inputChannels}/{outputChannels} channels)"

    def createInputStream(self) -> bool:
        '''Creates an input stream if it doesn't already exist.
           Returns True if stream already exists or was created successfuly, False otherwise.'''
        if self.inputStream:
            return True
        try:
            self.inputStream = self.audio.open(self.sampleRate, self.numInputChannels, self.audioFormat, True, False, self.inputDevice, None, self.chunkSize)
            return True
        except Exception as e:
            self.logger.warning(f"Error creating input stream: {(type(e).__name__)}: {e}")
            return False

    def createOutputStream(self) -> bool:
        '''Creates an output stream if it doesn't already exist.
           Returns True if stream already exists or was created successfuly, False otherwise.'''
        if self.outputStream:
            return True
        try:
            self.outputStream = self.audio.open(self.sampleRate, self.numOutputChannels, self.audioFormat, False, True, None, self.outputDevice, self.chunkSize)
            return True
        except Exception as e:
            self.logger.warning(f"Error creating output stream: {(type(e).__name__)}: {e}")
            return False

    def closeInputStream(self) -> None:
        if self.inputStream:
            self.inputStream.close()
            self.inputStream = None

    def closeOutputStream(self) -> None:
        if self.outputStream:
            self.outputStream.close()
            self.outputStream = None

    def readAudioData(self) -> Optional[bytes]:
        '''Reads a chunk of audio data from the input stream.
           Returns the retrieved data if successful, None otherwise.'''
        if not self.createInputStream():
            return None
        try:
            assert self.inputStream
            data = self.inputStream.read(self.chunkSize)
            return data
        except Exception as e:
            # Note: IOError: [Errno Input overflowed] -9981 often occurs when running under debugger
            # Note: IOError: [Errno Unanticipated host error] -9999 occurs when audio device is removed (cable unplugged)
            # Note: After 5-10 occurences of the above exception system hangs, so stream re-create seems necessary
            self.logger.warning(f"Audio input error: {(type(e).__name__)}: {e}")
            self.closeInputStream()
            self.saveSample()
            return None

    def writeAudioData(self, data: bytes) -> bool:
        '''Writes a chunk of audio data to the output stream.
           Returns True if successful, False otherwise.'''
        if not self.createOutputStream():
            return False
        try:
            assert self.outputStream
            self.outputStream.write(data)
            return True
        except Exception as e:
            self.logger.warning(f"Audio output error: {(type(e).__name__)}: {e}")
            self.closeOutputStream()
            return False

    def saveSample(self) -> bool:
        '''Saves the curent sample to the audio file.
           If the file does not exists, it is created.
           If the sample length is not equal to the self.sampleLength value, it means, we've cut
           the silence at the end of the sample, so it's the end of the file and it should be closed.
           The function returns True on success or if the recording is off, False otherwise.'''
        if not self.recording:
            return True
        try:
            if self.sampleLength:
                finalSample = True
            else:
                # If sampleLength wasn't set manualy, all the sample is saved.
                # It means the recording isn't over yet.
                self.sampleLength = len(self.sample)
                finalSample = False

            self.audioFileLength += self.sampleLength
            recordLength = (float(self.audioFileLength) / self.outputSecondSize)

            if recordLength > self.minRecordingLength: # The save-to-file process starts only when the sample is long enough
                if not self.audioFile: # Creating the file if necessary
                    self.audioFile = waveOpen(self.fileName, 'wb')
                    assert self.audioFile
                    self.audioFile.setnchannels(self.numOutputChannels)
                    self.audioFile.setsampwidth(self.audioBytes)
                    self.audioFile.setframerate(self.sampleRate)

                self.audioFile.writeframes(self.sample[:self.sampleLength]) # Removing extra silence at the end, if needed

                self.sample = b''
                self.sampleLength = 0

                if finalSample or not self.inLoop:
                    self.recording = False
                    self.audioFile.close()
                    self.audioFile = None
                    self.logger.info(f"Recording finished, max volume {self.localMaxVolume:.2f}%, {recordLength:.1f} seconds")

                return True
            if finalSample or not self.inLoop:
                self.recording = False
                self.logger.info(f"Recording discarded as it's too short ({recordLength:.1f} seconds)")
            else:
                self.audioFileLength -= self.sampleLength # If the sample is short we do not operate with it, so param changes should be undone
            return True
        except Exception as e:
            self.logger.warning(f"File output error: {(type(e).__name__)}: {e}")
            return False

    def run(self) -> None:
        '''Runs main audio processing loop.'''
        self.audioFile = None
        self.sampleLength = 0
        self.audioFileLength = 0
        self.inLoop = True
        self.recording = False
        self.quitAfterRecording = False
        self.lastSecondVolumes = [0.0] * self.chunksInSecond
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
                    data = b''.join(data[i : i + self.audioBytes] for i in range((self.channel - 1) * self.audioBytes, len(data), self.numInputChannels * self.audioBytes))
                assert len(data) == self.outputBlockSize

                if self.monitor: # Provide monitor output
                    self.writeAudioData(data)

                # Gathering volume statistics
                volume = (mean(abs(cast(int, unpack(self.packFormat, data[i : i + self.audioBytes])[0])) for i in range(0, len(data), self.audioBytes)) * 100 + self.maxVolume // 2) / self.maxVolume # pylint: disable=old-division
                self.lastSecondVolumes[chunkInSecond] = volume # Logging the sound volume during the last second
                chunkInSecond = (chunkInSecond + 1) % self.chunksInSecond

                if volume >= self.volumeTreshold: # The chunk is loud enough
                    if not self.recording: # Start recording
                        # ToDo: check inputStream.get_time(), latency etc. to provide exact time stamp for file naming
                        self.fileName = strftime(self.fileNameFormat)
                        self.logger.info(f"{self.fileName} recording started")
                        self.recording = True
                        self.sample = b''
                        self.localMaxVolume = volume
                        self.audioFileLength = 0
                    elif volume > self.localMaxVolume:
                        self.localMaxVolume = volume
                    self.sampleLength = 0
                    chunksOfSilence = 0
                    self.sample += data
                    self.saveSample()
                elif self.recording: # Check for stop recording
                    self.sample += data
                    chunksOfSilence += 1
                    if not self.sampleLength and chunksOfSilence > self.chunksOfFadeout: # Enough silence for a trail
                        self.sampleLength = len(self.sample) # Removing extra silence at the end
                    if chunksOfSilence > self.chunksToStop: # Enough silence to stop recording
                        self.saveSample() # Stopping recording
                        if self.quitAfterRecording:
                            self.inLoop = False
        except Exception as e:
            self.logger.warning(f"Processing error: {(type(e).__name__)}: {e}")
        except KeyboardInterrupt:
            self.logger.warning("Ctrl-C detected at input, exiting")
        self.inLoop = False
        self.saveSample()
        self.closeInputStream()
        self.closeOutputStream()
        self.logger.info("Done")

    def commandConsole(self) -> None:
        '''Runs in a separate thread to provide a command line operation adjustments.'''
        try:
            while self.inLoop:
                inp = input().split(' ')
                command = inp[0].lower()
                if 'help'.startswith(command):
                    print("""\nAvailable console commands (first letter is enough):
Help               - Show this information
EXit/Quit          - Exit the program immediately
Last               - Exit the program after completion of the current file
Volume             - Print the current mean volume level
Monitor [on/off]   - Show or toggle monitor status
Threshold [value]  - Show or set the volume threshold level\n""")
                elif 'exit'.startswith(command) or command == 'x' or 'quit'.startswith(command):
                    self.logger.info("Exiting")
                    self.inLoop = False
                elif 'volume'.startswith(command):
                    print(f"{mean(self.lastSecondVolumes):.2f}%") # Using print for non-functional logging
                elif 'monitor'.startswith(command):
                    if len(inp) < 2:
                        print(f"Monitor is {'ON' if self.monitor else 'OFF'}") # Using print for non-functional logging
                    else:
                        self.monitor = inp[1].lower().strip() in ('true', 'yes', 'on', '1')
                        self.logger.info(f"Monitor is set to {'ON' if self.monitor else 'OFF'}")
                elif 'last'.startswith(command):
                    if self.recording:
                        self.quitAfterRecording = True
                        self.logger.info("Going to exit after the end of the recording")
                    else:
                        self.logger.info("Exiting")
                        self.inLoop = False
                elif 'threshold'.startswith(command):
                    if len(inp) < 2:
                        print(f"Current volume treshold: {self.volumeTreshold:.2f}%") # Using print for non-functional logging
                    else:
                        try:
                            self.volumeTreshold = float(inp[1])
                            if not 0 <= self.volumeTreshold <= 100:
                                raise ValueError()
                            self.logger.info(f"New volume treshold: {self.volumeTreshold:.2f}%")
                        except ValueError:
                            print("Bad value, expected 0-100") # Using print for non-functional logging
        except EOFError:
            self.logger.warning("Console EOF detected")
        except Exception as e:
            self.logger.warning(f"Console error: {type(e).__name__}: {e}\n{format_exc()}")
            self.inLoop = False
        except KeyboardInterrupt:
            self.logger.warning("Ctrl-C detected at console, exiting")
            self.inLoop = False

    def sigTerm(self, _signum: int, _frame: FrameType) -> None:
        '''SIGTERM handler.'''
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

def usage(error: Optional[Exception] = None) -> NoReturn:
    '''Prints usage information (preceded by optional error message) and exits with code 2.'''
    print(f"{TITLE}\n")
    if error:
        print(f"Error: {error}\n")
    print("Usage: python RadioChronicle.py [-c configFileName] [-h]")
    print(f"\t-c --config <filename>   Configuration file to use, defaults to {DEFAULT_CONFIG_FILE_NAME}")
    print("\t-h --help                Show this help message")
    sysExit(2)

def main() -> None:
    RadioChronicle().run()

if __name__ == '__main__':
    main()
