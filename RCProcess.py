#!/usr/bin/python3
#
# RCProcess.py
#
# A part of RadioChronicle project
#
# by Vladimir Yashunsky (vladimir.yashunsky@gmail.com)
# and Vasily Zakharov (vmzakhar@gmail.com)
# https://github.com/jolaf/radiochronicle
#
# Version 0.6
#
# Requires webcolors: http://pypi.python.org/pypi/webcolors/
#
from configparser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime, timedelta
from enum import Enum, EnumMeta
from functools import total_ordering
from getopt import getopt
from glob import glob
from inspect import getmembers
from itertools import chain
from logging import getLogger, StreamHandler, NOTSET
from logging.config import fileConfig
from re import compile as reCompile
from signal import signal, SIGTERM
from statistics import mean
from struct import unpack
from sys import argv, exit as sysExit
from _thread import start_new_thread
from time import sleep
from types import FrameType
from typing import cast, Any, Callable, ClassVar, Final, FrozenSet, Iterable, Iterator, Mapping, NoReturn, Optional, Sequence, Tuple, Type, TypeVar
from wave import open as waveOpen, Wave_read as WaveReader

try:
    from webcolors import name_to_rgb, hex_to_rgb, rgb_percent_to_rgb, IntegerRGB, PercentRGB
except ImportError as e:
    raise ImportError(f"{type(e).__name__}: {e}\nPlease install WebColors: http://pypi.python.org/pypi/webcolors") from e

TITLE: Final[str] = "RCProcess v0.2  htts://github.com/jolaf/radiochronicle"
DEFAULT_CONFIG_FILE_NAME: Final[str] = 'rcp.conf'

SEPARATOR = reCompile(r'[\s,;]+')

RGB_PATTERN = reCompile(r'^(?:rgb)?\s*\(\s*(.*?)\s*\)$')

def parseHTMLColor(color: str) -> IntegerRGB:
    '''Converts all possible color specifications to (R, G, B) tuple.'''
    color = color.strip()
    try:
        return name_to_rgb(color)
    except ValueError:
        pass
    if len(color) in (4, 7):
        try:
            return hex_to_rgb(color)
        except ValueError:
            pass
    if len(color) in (3, 6):
        try:
            return hex_to_rgb('#' + color)
        except ValueError:
            pass
    try:
        m = RGB_PATTERN.match(color)
        if not m:
            raise ValueError()
        str3 = tuple(SEPARATOR.split(m.group(1)))
        if len(str3) != 3:
            raise ValueError
        try:
            for t in str3:
                if t[-1] != '%':
                    break
                value = float(t[:-1])
                if value < 0 or value > 100:
                    break
            else:
                return rgb_percent_to_rgb(cast(PercentRGB, str3))
        except ValueError:
            pass
        try:
            int3 = tuple(int(i) for i in str3)
            for value in int3:
                if value < 0 or value > 255:
                    break
            else:
                return cast(IntegerRGB, int3)
        except Exception:
            pass
    except Exception:
        pass
    raise ValueError(f"Couldn't identify color: {color}")

def strftime(timeFormat: str, time: datetime) -> str:
    '''Formats time to a string, supporting additional format %: which is formatted
       to a ':' in the first half of a second, and to a ' ' in the second half.'''
    return time.strftime(timeFormat.replace('%:', ':' if time.microsecond < 500000 else ' '))

def strptime(timeFormat: str, time: str) -> datetime:
    '''Parses a time string, supporting additional format %:
       which stands for either ':' or ' ' in the input.'''
    try:
        return datetime.strptime(time, timeFormat.replace('%:', ':'))
    except ValueError:
        return datetime.strptime(time, timeFormat.replace('%:', ' '))

def validateTimeFormat(timeFormat: str) -> bool:
    '''Returns True if the specified time format is valid, False otherwise.'''
    try:
        strftime(timeFormat, datetime.today())
        return True
    except ValueError:
        return False

def validateTime(timeFormat: str, time: str) -> bool:
    '''Returns True if the specified time is correctly formatted
       according to the specified format, False otherwise.'''
    try:
        strptime(timeFormat, time)
        return True
    except ValueError:
        return False

def diffTime(startTime: str, endTime: str) -> str:
    '''Reverse of relativeTime().
       Removes the common beginning from the endTime.'''
    nd = 0
    for i in range(min(len(startTime), len(endTime))):
        if not endTime[i].isdigit():
            if startTime[:i + 1] == endTime[:i + 1]:
                nd = i + 1
            else:
                break
    return endTime[nd:]

def relativeTime(startTime: str, endTime: str) -> str:
    '''Reverse of diffTime().
       Left-pads endTime to the length of startTime using startTime's beginning.'''
    assert len(startTime) >= len(endTime)
    return startTime[:len(startTime) - len(endTime)] + endTime

TConfigurable = TypeVar('TConfigurable', bound = 'Configurable')

class Configurable:
    '''Describes an object configurable with a ConfigParser section.'''

    Unknowns: ClassVar[EnumMeta] = Enum('Unknowns', ('INCLUDE', 'DISCARD', 'DENY', 'IGNORE'))

    _BOOLEAN_STATES: ClassVar[Mapping[str, bool]] = ConfigParser.BOOLEAN_STATES

    _CONVERTERS: ClassVar[Mapping[Type[Any], Callable[[str], Any]]] = { # Mapping a type to a callable returning a value of that type
                    bool: lambda value: Configurable._BOOLEAN_STATES[value.lower()],
                   float: float,
                     int: int,
                     str: lambda value: value,
              type(None): lambda value: value }

    _EXPECTS: ClassVar[Mapping[Type[Any], str]] = { # Mapping a type to a human-readable string describing possible values of that type
                    bool: f"{'/'.join(sorted(key for (key, value) in _BOOLEAN_STATES.items() if value))} or "
                          f"{'/'.join(sorted(key for (key, value) in _BOOLEAN_STATES.items() if not value))}",
                   float: "a float",
                     int: "an integer" }

    SECTION_PREFIX: ClassVar[str] = '' # Can be overriden in subclasses to automatically cut out the standard section name prefix for that class.

    SECTION_SEPARATORS: ClassVar[str] = ' -_' # Can be overriden in subclasses

    def __init__(self, config: ConfigParser, section: str, *, unknowns: Unknowns = Unknowns.DENY, raw: bool = False, vars: Optional[Mapping[str, str]] = None, defaults: Optional[Mapping[str, Any]] = None) -> None: # pylint: disable=redefined-builtin
        if defaults:
            for (name, value) in defaults.items():
                setattr(self, name, value)
        members = ((field, default) for (field, default) in getmembers(self, lambda member: not callable(member)) if field[0].islower())
        fields = {field.lower(): (field, type(default)) for (field, default) in members}
        sections = self._getSections(config, section, raw, vars)
        options = (tuple((option, (section, value.strip())) for (option, value) in config.items(section, raw, vars) if option != 'inherit') for section in sections)
        self.config = dict(chain.from_iterable(options))
        self.section = section
        self.name = section[len(self.SECTION_PREFIX):].strip(self.SECTION_SEPARATORS) if self.SECTION_PREFIX else section
        for (option, (section, value)) in self.config.items(): # pylint: disable=redefined-argument-from-local
            try:
                (field, typ) = fields[option]
                try:
                    setattr(self, field, self._valueParser(typ, value))
                except Exception as e:
                    raise ValueError(f"Bad value for [{section}].{field}: '{value}', must be {self._getExpects(typ)}") from e
            except KeyError as e:
                if unknowns == self.Unknowns.INCLUDE:
                    setattr(self, option, value)
                elif unknowns == self.Unknowns.DISCARD:
                    del self.config[option]
                elif unknowns == self.Unknowns.DENY:
                    raise ValueError(f"Unknown option [{section}].{option}") from e

    @classmethod
    def loadSections(cls: Type[TConfigurable], config: ConfigParser, **kwargs: Any) -> Sequence[TConfigurable]:
        '''Creates Configurable objects of this class from all sections in the specified config having the proper prefix.'''
        return tuple(cls(config, section, **kwargs) for section in sorted(config.sections()) if section.startswith(cls.SECTION_PREFIX))

    @staticmethod
    def _getSections(config: ConfigParser, section: str, raw: bool = False, vars: Optional[Mapping[str, str]] = None, previousSections: FrozenSet[str] = frozenset()) -> Iterator[str]: # pylint: disable=redefined-builtin
        '''Resursively retrieves list of sections considering inheritance.'''
        if not config.has_section(section):
            raise Exception(f"Section [{section}] not found")
        ps: FrozenSet[str] = previousSections | {section}
        if config.has_option(section, 'inherit'):
            for s in SEPARATOR.split(config.get(section, 'inherit', raw = raw, vars = vars)):
                if s in ps:
                    raise ValueError(f"Inheritance recursion detected: {section} -> {s}")
                if not config.has_section(s):
                    raise ValueError(f"Inherited section not found: {section} -> {s}")
                yield from Configurable._getSections(config, s, raw, vars, ps)
        yield section

    @classmethod
    def _getSubclasses(cls: Type[TConfigurable], includeFilter: Callable[[Type[TConfigurable]], Any] = lambda cls: True) -> Iterator[Type[TConfigurable]]:
        for subClass in cls.__subclasses__():
            if includeFilter(subClass):
                yield subClass
            yield from subClass._getSubclasses() # pylint: disable=protected-access

    @classmethod
    def loadSectionsWithSubclasses(cls: Type[TConfigurable], config: ConfigParser, includeFilter: Callable[[Type[TConfigurable]], Any] = lambda cls: True, **kwargs: Any) -> Sequence[TConfigurable]:
        return tuple(chain.from_iterable(subclass.loadSections(config, **kwargs) for subclass in cls._getSubclasses(includeFilter)))

    @staticmethod
    def _valueParser(typ: Type[Any], value: str) -> Any:
        if not issubclass(typ, Enum):
            return Configurable._CONVERTERS.get(typ, typ)(value)
        value = value.strip().upper()
        values = tuple(key.upper() for key in typ.__members__)
        for i in range(1, len(value) + 1):
            matches = tuple(v for v in values if value[:i] == v[:i])
            if len(matches) == 1:
                return typ[matches[0]]
            if not matches:
                break
        raise ValueError(f"Bad value for type {typ.__name__}: {repr(value)}")

    @staticmethod
    def _getExpects(typ: Type[Any]) -> str:
        return '/'.join(e.name.upper() for e in typ) if issubclass(typ, Enum) else Configurable._EXPECTS.get(typ) or f'suitable for constructing class {typ.__name__}'

@total_ordering
class TimePoint:
    START = 'START'	# If timepoint is inside the file, start from the beginning of that file
    END = 'END'		# If timepoint is inside the file, end by the end of that file
    NEXT = 'NEXT'	# If timepoint is inside the file, start from the beginning of the next file
    PREV = 'PREV'	# If timepoint is inside the file, end by the end of the previous file
    CUT = 'CUT'		# If timepoint is inside the file, start from (end by) exactly that point
    SKIP = 'SKIP'	# If timepoint is outside any file, start from the beginning (end by the end) of the next (previous) file
    KEEP = 'KEEP'	# If timepoint is outside any file, start from (end by) exactly that point (keep silence)

    time: datetime
    fromThis = True	# If time is set, exactly one of fromStart/toEnd, fromNext/toPrev and cut must be True
    fromNext = False
    keep = False

    def __init__(self, isStart, time, timeFormat, location = None) -> None:
        self.isStart = isStart
        if isinstance(time, datetime):
            self.time = time
            return
        args = time.split()
        if not args:
            raise ValueError(f"Date not specified{f' at {location}' if location else ''}")
        time = args[0]
        try:
            self.time = strptime(timeFormat, time)
        except ValueError:
            raise ValueError(f"Format '{timeFormat}' is not matched by the data{f' at {location}' if location else ''}: '{time}'")
        inSet = outSet = False
        for arg in (arg.upper() for arg in args[1:]):
            if arg in (self.START, self.END, self.NEXT, self.PREV, self.CUT):
                if arg in (self.END, self.PREV) if isStart else (self.START, self.NEXT):
                    raise ValueError(f"Unexpected {'start' if isStart else 'end'} time point specifier{f' at {location}' if location else ''}: '{arg}', allowed {', '.join((self.START, self.NEXT) if isStart else (self.END, self.PREV) + (self.CUT, self.SKIP, self.KEEP))}")
                if inSet:
                    raise ValueError(f"Too many time point specifiers{f' at {location}' if location else ''}: '{arg}'")
                if arg in (self.NEXT, self.PREV):
                    self.fromThis = False
                    self.fromNext = True
                elif arg == self.CUT:
                    self.fromThis = False
                inSet = True
            elif arg in (self.SKIP, self.KEEP):
                if outSet:
                    raise ValueError(f"Too many time point specifiers{f' at {location}' if location else ''}: '{arg}'")
                self.keep = (arg == self.KEEP)
                outSet = True
            else:
                raise ValueError(f"Unknown time point specifier{f' at {location}' if location else ''}: '{arg}'")

    def __lt__(self, other: Any) -> bool: # @total_ordering does the rest
        if isinstance(other, TimePoint):
            return self.time < other.time
        return NotImplemented

    def match(self, startTime: datetime, endTime: datetime) -> Optional[datetime]:
        if endTime < self.time: # before
            return None if self.isStart else (self.time if self.keep and self.time != datetime.max else endTime)
        if startTime <= self.time: # inside
            return (startTime if self.isStart else endTime) if self.fromThis else None if self.fromNext else self.time
        # after
        return (self.time if self.keep and self.time != datetime.min else startTime) if self.isStart else None

class InputFile:
    '''Holds information on the input audio file and methods to access it.'''
    N_CHANNELS_NAMES: Final[ClassVar[Mapping[int, str]]] = { 1: 'Mono', 2: 'Stereo' }

    # ToDo: Change Audio library to support mp3 etc.
    def __init__(self, fileName: str, startTime: str) -> None:
        self.fileName = fileName
        self.startTime = startTime
        self.wave: WaveReader = waveOpen(fileName)
        (self.nChannels, self.audioBytes, self.sampleRate, self.nSamples, self.compressionType, self.compressionName) = self.wave.getparams()
        if self.compressionType != 'NONE':
            raise ValueError(f"Unknown compression type '{self.compressionType}' for file {fileName}")
        self.audioBits = self.audioBytes * 8
        self.sampleSize = self.nChannels * self.audioBytes
        self.length = timedelta(seconds = float(self.nSamples) / self.sampleRate)
        self.endTime = self.startTime + self.length
        # ToDo: Maybe round length right in the fields?
        self.printLength = self.length + timedelta(microseconds = (self.length.microseconds + 5000) // 10000 * 10000 - self.length.microseconds)
        self.name = f"{self.fileName} {self.sampleRate}Hz/{self.audioBits}-bit/{self.N_CHANNELS_NAMES.get(self.nChannels) or f'{self.nChannels}-channels'} {self.printLength}"

    def contains(self, time: str) -> bool:
        return self.startTime <= time < self.endTime

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"InputFile({self.fileName!r}, {self.startTime!r})"

    def read(self, nSamples: int) -> bytes:
        ret = self.wave.readframes(nSamples)
        assert len(ret) == nSamples * self.sampleSize
        return ret

    def rewind(self) -> None:
        self.wave.rewind()

    def close(self) -> None:
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

    def __init__(self, config: ConfigParser, section: str) -> None:
        super().__init__(config, section)
        self.section = section
        if not validateTimeFormat(self.fileNameFormat):
            raise ValueError(f"Bad file name format at [{section}].fileNameFormat: '{self.fileNameFormat}'")
        self.startPoint = TimePoint(True, *(self.startPoint, f'[{section}].startPoint') if self.startPoint else datetime.min)
        self.endPoint = TimePoint(False, *(self.endPoint, f'[{section}].endPoint') if self.endPoint else datetime.max)
        if self.channels:
            try:
                self.channels = self.CHANNEL_NUMBERS[self.channels.upper()]
            except KeyError:
                raise ValueError(f"Bad channels specification at [{section}].channels: '{self.channels}'")
        else:
            self.channels = self.STEREO

    @staticmethod
    def getFileTimes(fileNameFormat: str) -> Iterator[Tuple[str, str]]:
        for fileName in glob(sub('%.', '*', fileNameFormat)):
            try:
                yield (strptime(fileNameFormat, fileName), fileName)
            except ValueError:
                pass

    @staticmethod
    def preSortFiles(files: Iterable[Tuple[str, str]], startTime: str, endTime: str) -> Iterator[Tuple[str, str]]:
        assert startTime <= endTime
        started = False
        prev: Optional[Tuple[str, str]] = None
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

    def openFiles(self, files: Iterable[Tuple[str, str]]) -> Iterator[InputFile]:
        prev: Optional[InputFile] = None
        self.startTime: Optional[datetime] = None
        self.endTime: Optional[datetime] = None
        for (time, fileName) in files:
            f = InputFile(fileName, time)
            if prev and f.startTime < prev.endTime:
                raise ValueError(f"Files intersect in time: {f} start < {prev} end")
            prev = f
            if not self.startTime:
                self.startTime = self.startPoint.match(f.startTime, f.endTime)
                if not self.startTime:
                    continue
            endTime = self.endPoint.match(f.startTime, f.endTime)
            if not endTime:
                return
            yield f
            self.endTime = endTime

    def load(self) -> None:
        self.files = tuple(self.openFiles(self.preSortFiles(self.getFileTimes(self.fileNameFormat), self.startPoint.time, self.endPoint.time)))
        if not self.files:
            raise ValueError(f"No files found matching [{self.section}].fileNameFormat: '{self.fileNameFormat}'")

class Output(Configurable):

    def __init__(self, config: ConfigParser, section: str) -> None:
        super().__init__(config, section)

class Visual(Configurable):

    TAG = 'type'

    def __new__(cls, config: ConfigParser, section: str) -> Visual:
        if cls != Visual:
            raise ValueError("Visual.__new__() can only be used to create Visual instances")
        scn = Configurable(config, section, Configurable.IGNORE, defaults = {Visual.TAG: None}).config.get(Visual.TAG)
        if not scn:
            raise ValueError(f"Can't create Visual: section [{section}] doesn't contain option '{Visual.TAG}'")
        (section, className) = scn
        cls = globals().get(className[:1].upper() + className[1:])
        if not cls or not isinstance(cls, type) or not issubclass(cls, Visual):
            raise ValueError(f"Unknown Visual type at [{section}].{Visual.TAG}: '{className}'")
        return Configurable.__new__(cls)

    def __init__(self, config: ConfigParser, section: str) -> None:
        super().__init__(config, section)

class Text(Visual):
    def __init__(self, config: ConfigParser, section: str) -> None:
        super().__init__(config, section)

class RCProcess: # pylint: disable=too-many-instance-attributes
    # Default parameter values
    logger = None

    def __init__(self) -> None: # pylint: disable=too-many-statements, too-complex
        '''Fully constructs class instance, including reading configuration file and configuring audio devices.'''
        try: # Reading command line options
            configFileName = DEFAULT_CONFIG_FILE_NAME
            (options, args) = getopt(argv[1:], 'c:h', ('config=', 'help')) # args in unused # pylint: disable=W0612
            for (option, value) in options:
                if option in ('-c', '--config'):
                    configFileName = value.strip()
                else:
                    usage()
        except Exception as e:
            usage(e)
        try: # Reading config file and configuring logging
            config = ConfigParser()
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
            sysExit(1)
        # Above this point, use print for diagnostics
        # From this point on, we have self.logger to use instead
        self.logger.info(TITLE)
        self.logger.info(f"Using {configFileName}")
        print() # Empty line to console only
        try: # Applying configuration
            self.inputs = tuple(Input(config, section) for section in sorted(section for section in config.sections() if section.startswith('input')))
            if not self.inputs:
                raise Exception("No [input*] sections found")
            value = None
            try:
                # ToDo: create wrappers for getting config parameters, for RC also
                section = 'general'
                try:
                    self.fileNameFormat = config.get(section, 'fileNameFormat').strip()
                    # ToDo: check that files exist
                except NoOptionError:
                    pass
                try:
                    self.startDate = config.get(section, 'startDate')
                    # ToDo: parse date
                except NoOptionError:
                    pass
                try:
                    self.endDate = config.get(section, 'endDate')
                    # ToDo: parse date
                except NoOptionError:
                    pass
                try:
                    self.video = config.getboolean(section, 'video')
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].video: '{config.get(section, 'video')}', must be 1/yes/true/on or 0/no/false/off") from e
                try:
                    self.audio = config.getboolean(section, 'audio')
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].audio: '{config.get(section, 'audio')}', must be 1/yes/true/on or 0/no/false/off") from e
            except NoSectionError:
                pass
            try:
                section = 'video'
                # ToDo: specify video format somehow
                try:
                    value = config.get(section, 'width')
                    self.width = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].width: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'height')
                    self.height = int(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].height: '{value}', must be an integer") from e
                try:
                    value = config.get(section, 'fps')
                    self.fps = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].fps: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'background')
                    self.background = parseHTMLColor(value)
                    # ToDo: think about proper color representation
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].background: '{value}', must be a #rrggbb string") from e
                try:
                    self.previewVideo = config.getboolean(section, 'preview')
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].preview: '{config.get(section, 'preview')}', must be 1/yes/true/on or 0/no/false/off") from e
            except NoSectionError:
                pass
            try:
                section = 'audio'
                try:
                    value = config.get(section, 'audioBits')
                    self.audioBits = int(value)
                    # ToDo: check it further
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
                    value = config.get(section, 'channels')
                    self.channels = int(value)
                    # ToDo: check it's positive
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].channels: '{value}', must be an integer") from e
                try:
                    self.previewAudio = config.getboolean(section, 'preview')
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].preview: '{config.get(section, 'preview')}', must be 1/yes/true/on or 0/no/false/off") from e
            except NoSectionError:
                pass
            try:
                section = 'tuning'
                try:
                    value = config.get(section, 'maxPauseLength')
                    self.maxPauseLength = float(value)
                    self.replacePauseLength = self.maxPauseLength
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].maxPauseLength: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'replacePauseLength')
                    self.replacePauseLength = float(value)
                    # ToDo: check replacePauseLength <= maxPauseLength
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].replacePauseLength: '{value}', must be a float") from e
                try:
                    value = config.get(section, 'trailLength')
                    self.trailLength = float(value)
                except NoOptionError:
                    pass
                except ValueError as e:
                    raise ValueError(f"Bad value for [{section}].trailLength: '{value}', must be a float") from e
            except NoSectionError:
                pass

            # Validating configuration parameters
            if not self.fileNameFormat:
                raise ValueError("Bad value for fileNameFormat: must be not empty")
            if not 0 <= self.volumeTreshold <= 100:
                raise ValueError(f"Bad value for volumeTreshold: {self.volumeTreshold:2f}, must be 0-100")
            if self.maxPauseLength < 0:
                self.maxPauseLength = 0
            if self.minRecordingLength < 0:
                self.minRecordingLength = 0
            if self.trailLength < 0:
                self.trailLength = 0
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
                self.channel = int(channel)
                if self.channel <= 0:
                    self.channel = None # Exception will be thrown below
            except ValueError:
                self.channel = CHANNEL_NUMBERS.get(channel.strip().upper()) # Would be None if not found
            if self.channel == None:
                raise ValueError(f"Bad value for channel: {channel}, must be LEFT/RIGHT/STEREO/ALL/MONO or a number of 1 or more")

            # Accessing audio devices
            try:
                if self.inputDevice != None:
                    inputDeviceInfo = self.audio.get_device_info_by_index(self.inputDevice)
                    self.logger.info(f"Using input device {self.deviceInfo(inputDeviceInfo, False)}")
                else:
                    inputDeviceInfo = self.audio.get_default_input_device_info()
                    self.logger.info(f"Using default input device {self.deviceInfo(inputDeviceInfo, False)}")
            except ValueError:
                raise ValueError(f"{f'Input device {self.inputDevice}' if self.inputDevice != None else 'Default input device'} is not in fact an input device")
            except IOError as e:
                raise IOError(f"Can't access {f'input device {self.inputDevice}' if self.inputDevice != None else 'default input device'}: {e}")
            try:
                if self.outputDevice != None:
                    outputDeviceInfo = self.audio.get_device_info_by_index(self.outputDevice)
                    self.logger.info(f"Using output device {self.deviceInfo(outputDeviceInfo, True)}")
                else:
                    outputDeviceInfo = self.audio.get_default_output_device_info()
                    self.logger.info(f"Using default output device {self.deviceInfo(outputDeviceInfo, True)}")
            except ValueError:
                raise ValueError(f"{f'Output device {self.outputDevice}' if self.outputDevice != None else 'Default output device'} is not in fact an output device")
            except IOError as e:
                raise IOError(f"Can't access {f'output device {self.outputDevice}' if self.outputDevice != None else 'default output device'}: {e}")
            print() # Empty line to console only

            # Calculating derivative paratemers
            self.numInputChannels = 1 if self.channel == MONO else inputDeviceInfo['maxInputChannels']
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
            self.logger.info(f"Volume threshold {self.volumeTreshold:2f}, max pause {self.maxPauseLength:1f} seconds, min recording length {self.minRecordingLength:1f} seconds, trail {self.trailLength:1f} seconds")
            self.logger.info(f"Monitor is {'ON' if self.monitor else 'OFF'}")
            print("Type 'help' for console commands reference") # Using print for non-functional logging
            print() # Empty line to console only
        except Exception as e:
            self.logger.error(f"Configuration error: {e}")
            sysExit(1)

    def run(self) -> None:
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
                    data = ''.join(data[i : i + self.audioBytes] for i in range((self.channel - 1) * self.audioBytes, len(data), self.numInputChannels * self.audioBytes))
                assert len(data) == self.outputBlockSize

                if self.monitor: # Provide monitor output
                    self.writeAudioData(data)

                # Gathering volume statistics
                volume = (mean(abs(unpack(self.packFormat, data[i : i + self.audioBytes])[0]) for i in range(0, len(data), self.audioBytes)) * 100 + self.maxVolume // 2) / self.maxVolume
                self.lastSecondVolumes[chunkInSecond] = volume # Logging the sound volume during the last second
                chunkInSecond = (chunkInSecond + 1) % self.chunksInSecond

                if volume >= self.volumeTreshold: # The chunk is loud enough
                    if not self.recording: # Start recording
                        # ToDo: check inputStream.get_time(), latency etc. to provide exact time stamp for file naming
                        self.fileName = strftime(self.fileNameFormat)
                        self.logger.info(f"{self.fileName} recording started")
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
        except Exception as e:
            self.logger.warning(f"Processing error: {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            self.logger.warning("Ctrl-C detected, exiting")
        self.inLoop = False
        self.logger.info("Done")

    def sigTerm(self, _signum: int, _frame: FrameType) -> None:
        '''SIGTERM handler.'''
        self.logger.warning("SIGTERM caught, exiting")
        self.inLoop = False

def usage(error: Optional[Exception] = None) -> NoReturn:
    '''Prints usage information (preceded by optional error message) and exits with code 2.'''
    print(f"{TITLE}\n")
    if error:
        print(f"Error: {error}\n")
    print("Usage: python RCProcess.py [-c configFileName] [-h]")
    print(f"\t-c --config <filename>   Configuration file to use, defaults to {DEFAULT_CONFIG_FILE_NAME}")
    print("\t-h --help                Show this help message")
    sysExit(2)

def main() -> None:
    RCProcess().run()

if __name__ == '__main__':
    main()
