#
# RadioChronicle
#
# Version 0.3
#
# Requires PyAudio: http://people.csail.mit.edu/hubert/pyaudio/
#

import struct
import sys
import thread
import wave

from time import strftime

from pyaudio import *

# Defines
PACK_FORMATS = { 8 : 'b', 16 : '<h', 32 : '<i' }

STEREO = 0
LEFT = 1
RIGHT = 2
MONO = 3

# Configurable parameters
FILENAME_FORMAT = '%Y%m%d-%H%M%S.wav'

SILENCE_SECONDS = 2	# The max time we wait for the transmission to restart
FADEOUT_SECONDS = 0.1	# The length of silence at the end of the recorded transmission
SILENCE_LEVEL = 0	# The max sound level (in percent) that is treated as silence

INPUT_DEVICE = None	# May be set to use non-default device
OUTPUT_DEVICE = None	# May be set to use non-default device
INPUT_CHANNELS = 2	# Number of channels in input device
AUDIO_BITS = 16		# Recording quantization in bits (8/16/24/32)
SAMPLE_RATE = 44100	# Sampling frequency

CHANNEL = MONO		# Input channel to monitor (LEFT, RIGHT, STEREO, MONO)

CHUNK_SIZE = 1024	# Number of frames to process at once
DIRECT_OUTPUT = False

# Configuration
assert AUDIO_BITS in (8, 16, 32)
AUDIO_BYTES = AUDIO_BITS / 8
MAX_VOLUME = 1 << (AUDIO_BITS - 1)

FRAMES_IN_SECOND = SAMPLE_RATE / CHUNK_SIZE
BLOCK_SIZE = CHUNK_SIZE * AUDIO_BYTES * (2 if CHANNEL == STEREO else 1)

audio = PyAudio()

AUDIO_FORMAT = audio.get_format_from_width(AUDIO_BYTES, False)
PACK_FORMAT = PACK_FORMATS[AUDIO_BITS]

#script-status parameters
inLoop = True 
recording = False
quitAfterRecording = False
lastSecondVolumes = [0] * FRAMES_IN_SECOND

def mean(iterable):
    if not iterable:
        return 0
    sum = 0
    for (n, i) in enumerate(iterable, 1):
        sum += i
    return sum / n

def printDeviceInfo():
    devices = tuple(audio.get_device_info_by_index(i) for i in xrange(p.get_device_count()))
    print "\nDetected input devices:"
    for device in filter(lambda d: d['maxInputChannels'], devices):
        print "%d: %s" % (device['index'], device['name'])
    print "\nDetected output devices:"
    for device in filter(lambda d: d['maxOutputChannels'], devices):
        print "%d: %s" % (device['index'], device['name'])

def commandConsole():
    global inLoop
    global DIRECT_OUTPUT
    global recording
    global quitAfterRecording
    global SILENCE_LEVEL
    global lastSecondVolumes
    global CHUNK_SIZE
    
    while inLoop:
        inp = raw_input().split(' ')
        command = inp[0]
        if inp[0] == "mean":
            print "%d%%" % mean(lastSecondVolumes)
        elif inp[0] == "quit":
            inLoop = False
        elif inp[0] == "output":
            if len(inp) < 2:
                print "Direct output status: %s" % DIRECT_OUTPUT
            else:
                DIRECT_OUTPUT = bool(inp[1])
                print "Direct output set to: %s" % DIRECT_OUTPUT
        elif inp[0] == "last":
            if recording:
                quitAfterRecording = True
                print "Recording the last track"
            else:
                inLoop = False
        elif inp[0] == "level":
            if len(inp) < 2:
                print "Current silence level: %d" % SILENCE_LEVEL
            else:
                try:
                    SILENCE_LEVEL = int(inp[1])
                    assert 0 <= SILENCE_LEVEL <= 100
                    print "New silence level: %d" % SILENCE_LEVEL
                except:
                    print "Bad value, expected 0-100"

thread.start_new_thread(commandConsole,())

stream = audio.open(format = AUDIO_FORMAT,
                channels = 1 if CHANNEL == MONO else 2,
                rate = SAMPLE_RATE,
                input = True,
                output = True, #this is for the direct sound output
                frames_per_buffer = CHUNK_SIZE)

chunksToStop = int(float(SAMPLE_RATE / CHUNK_SIZE) * SILENCE_SECONDS)
chunksOfFadeout = int(float(SAMPLE_RATE / CHUNK_SIZE) * FADEOUT_SECONDS)

print "Ready to work"

frameInSecond=0

while inLoop:
    # We've just recived a new part of sound data
    data = stream.read(CHUNK_SIZE)
    assert len(data) == BLOCK_SIZE

    if CHANNEL in (LEFT, RIGHT):
        data = ''.join(data[i : i + AUDIO_BYTES] for i in xrange(0 if CHANNEL == LEFT else AUDIO_BYTES, len(data), 2 * AUDIO_BYTES))

    if DIRECT_OUTPUT: # Provide monitor output
        # ToDo: Need separate output stream to allow only left/right channel output
        stream.write(data, CHUNK_SIZE)
    
    # Let's see if data contains something loud enough
    volume = (mean(abs(struct.unpack(PACK_FORMAT, data[i : i + AUDIO_BYTES])[0]) for i in xrange(0, len(data), AUDIO_BYTES)) * 100 + MAX_VOLUME / 2) / MAX_VOLUME
    lastSecondVolumes[frameInSecond] = volume # Logging the sound volume during the last second
    frameInSecond = (frameInSecond + 1) % FRAMES_IN_SECOND
                 
    if volume >= SILENCE_LEVEL:
        if not recording: # Start recording
            fileName = strftime(FILENAME_FORMAT)
            print "%s recording started" % fileName
            recording = True
            sample = ''
            maxVolume = volume
        elif volume > maxVolume:
            maxVolume = volume
        sample += data
        chunksOfSilence = 0
        sampleLength = -1
    elif recording: # Check for stop recording
        sample += data
        chunksOfSilence += 1
        if sampleLength < 0 and chunksOfSilence > min(chunksOfFadeout, chunksToStop):
            sampleLength = len(sample)
        if chunksOfSilence > chunksToStop: # Stopping recording
            recording = False
            wf = wave.open(fileName, 'wb')
            wf.setnchannels(2 if CHANNEL == STEREO else 1)
            wf.setsampwidth(AUDIO_BYTES)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(sample[:sampleLength]) # Removing extra silence at the end
            wf.close()
            sample = ''
            print "Recording finished, max volume %d, %d seconds" % (maxVolume, sampleLength / (SAMPLE_RATE * AUDIO_BYTES * (2 if CHANNEL == STEREO else 1)))
            if quitAfterRecording:
                inLoop = False

if recording: # Listening was interrupted in the middle of recording
    recording = False
    if sampleLength < 0:
        sampleLength = len(sample)
    wf = wave.open(fileName, 'wb')
    wf.setnchannels(2 if CHANNEL == STEREO else 1)
    wf.setsampwidth(AUDIO_BYTES)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(sample[:sampleLength]) # Removing extra silence at the end
    wf.close()  
    print "Recording finished, max volume %d, %d seconds" % (maxVolume, sampleLength / (SAMPLE_RATE * AUDIO_BYTES * (2 if CHANNEL == STEREO else 1)))

print "* done listening"

stream.close()
audio.terminate()
