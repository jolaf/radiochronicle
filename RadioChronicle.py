#requires http://people.csail.mit.edu/hubert/pyaudio/
import pyaudio
import wave
import sys
from time import strftime

RECORD_SECONDS = 60 #the life time of the programm. Actually, this variable shouldn't exist
                    #and the cycle should be written as an infinite loop, but I haven't yet understood
                    #how to do the user interface for breaking this loop.
SILENCE_SECONDS = 2 #The max time we wait for the speach to restart.
SILENCE_LEVEL = 50  #The level of sound that is concedered as silence


#the following params shouldn't be changed without the source adjustment
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
#######

isRecording = False

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True, #this is for the direct sound output
                frames_per_buffer = chunk)




stepToStop=RATE/chunk * SILENCE_SECONDS
stepsOfSilence=0

print "* start listening"

for i in range(0, RATE / chunk * RECORD_SECONDS):
    #we have just recived a new part of sound data
    data = stream.read(chunk)
    stream.write(data, chunk) #this wires the input to the output
    
    #let's see if it contains something loud enought
    summ=0;    
    for j in range(0,len(data)/2):
        lb=ord(data[2*j])
        hb=ord(data[2*j+1])
        summ=summ+abs(((hb*256+lb+32768) % 65536)-32768);
                
    if (((summ/chunk)>SILENCE_LEVEL) and (not(isRecording))):
        #if the signal is strong and we do not yet record, let's start
        WAVE_OUTPUT_FILENAME=strftime("%Y-%m-%d_%H.%M.%S")+".wav";
        print WAVE_OUTPUT_FILENAME+" record started"
        all = []
        all.append(data)
        isRecording=True
        stepsOfSilence=0
    
    if (((summ/chunk)>SILENCE_LEVEL) and (isRecording)):
        #if we allready are recording, let's proceede
        all.append(data)
        stepsOfSilence=0

    if (((summ/chunk)<=SILENCE_LEVEL) and (isRecording)):
        #if there is no more signal we do wait
        all.append(data)
        stepsOfSilence=stepsOfSilence+1
        
        if (stepsOfSilence>stepToStop):
            #oops, the silence was too long. it's over now
            isRecording=False
            data = ''.join(all)
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(data)
            wf.close()  
            all = []
            print "record finished. File length: "+str(len(data)/(RATE*2))+" seconds"          

# the listening is over
        
if isRecording:
    #hey, we have some unfinished job
    isRecording=False
    data = ''.join(all)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()  
    all = []
    print "record finished. File length: "+str(len(data)/(RATE*2))+" seconds"          

print "* done listening"

stream.close()
p.terminate()




