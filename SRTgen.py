#!/usr/bin/python3
#
# SRTgen.py
#
# A part of RadioChronicle project
#
# by Vasily Zakharov (vmzakhar@gmail.com)
# https://github.com/jolaf/radiochronicle
#
# Version 0.1
#
from datetime import datetime, timedelta
from re import compile as reCompile
from sys import argv
from typing import Sequence
from wave import open as waveOpen

SECOND = timedelta(seconds = 1)

FILENAME_PATTERN = reCompile(r'^(?:.*?[/\\])?(RC-.*?).wav$')

WAV_FILENAME_FORMAT = 'RC-%Y%m%d-%H%M%S.wav'
DISPLAY_TIME_FORMAT = '%a %H:%M:%S'
SRT_TIME_FORMAT = '%H:%M:%S,%f'

WEEKDAYS = ('ПН', 'ВТ', 'СР', 'ЧТ', 'ПТ', 'СБ', 'ВС')

def displayFormat(d: datetime) -> str:
    """Returns a string representation for a datetime in format `ДД HH:MM:SS`."""
    return d.strftime(DISPLAY_TIME_FORMAT.replace('%a', WEEKDAYS[d.weekday()]))

def srtFormat(d: datetime) -> str:
    """Returns a string representation for a datetime in format `HH:MM:SS,fff`.
       The idea is to make the `.srt` file length the same to Adobe Premiere as the `.wav` file length."""
    ms = d.microsecond // 1000
    if ms == 899:
        ms = 800 # Compatibility fix for Adobe Premiere
    return d.strftime(SRT_TIME_FORMAT.replace('%f', f'{ms:03}')) # Cut microseconds to milliseconds

def main(args: Sequence[str]) -> None:
    wavFileName = args[0]
    with waveOpen(wavFileName) as f:
        nChannels = f.getnchannels()
        assert nChannels == 1
        sampWidth = f.getsampwidth()
        assert sampWidth == 2
        frameRate = f.getframerate()
        assert frameRate == 44100
        nFrames = f.getnframes()
        assert nFrames
        assert f.getcomptype() == 'NONE'
        assert len(f.readframes(nFrames)) == nFrames * nChannels * sampWidth
    length = float(nFrames) / frameRate
    m = FILENAME_PATTERN.match(wavFileName)
    assert m # Should always match
    srtFileName = f'{m.group(1)}.srt'
    realTime = datetime.strptime(wavFileName, WAV_FILENAME_FORMAT)
    realTime += timedelta(hours = 2) # На «Обитаемых островах» ноутбук был настроен на неверную таймзону
    fileTime = datetime(1, 1, 1) # We don't actually use the date part
    srtEnd = fileTime + timedelta(seconds = length)
    maxN = int(length + 1 if length % 1 else length)
    with open(srtFileName, 'w', encoding = 'utf-8') as f: # Adobe Premiere only understands Russian in UTF-8 encoding
        for n in range(1, maxN + 1):
            nextTime = fileTime + SECOND
            f.write(f"{n}\n{srtFormat(fileTime)} --> {srtFormat(min(srtEnd, nextTime))}\n{displayFormat(realTime)}\n\n")
            realTime += SECOND
            fileTime = nextTime

if __name__ == '__main__':
    main(argv[1:])
