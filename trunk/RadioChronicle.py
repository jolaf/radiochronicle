#requires http://people.csail.mit.edu/hubert/pyaudio/
#version 0.2

import pyaudio
import wave
import sys
import struct
import thread
from time import strftime


#configurable parameters
SILENCE_SECONDS = 2 #The max time we wait for the speach to restart.
FADEOUT_SECONDS = 0.5 #The time for a good fade out
SILENCE_LEVEL = 100  #The level of sound that is concedered as silence
DIRECT_OUTPUT = False
FILENAME_FORMAT = "%Y%m%d-%H%M%S"

#unconfigurable parameters
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 #This one should be configurable, but it will be fixed later
RATE = 44100

#script-status parameters
inLoop = True 
isRecording = False
lastRecordingTrack = False

lastSecondArray=[]
lastSecondFramesCount=RATE/chunk
for i in range (0,lastSecondFramesCount):
    lastSecondArray.append(0)


def update_params():
    global inLoop
    global DIRECT_OUTPUT
    global isRecording
    global lastRecordingTrack
    global SILENCE_LEVEL
    global lastSecondArray
    global chunk
    
    while inLoop:
        inp=raw_input().split(" ")
        if inp[0]=="mean":
            print (sum(lastSecondArray)/(len(lastSecondArray)*chunk))
        if inp[0]=="quit":
            inLoop=False
        if inp[0]=="output":
            if len(inp)==1:
                print "Direct output status: "+str(DIRECT_OUTPUT)    
            else:
                if inp[1]=="True":
                    DIRECT_OUTPUT=True;
                if inp[1]=="False":
                    DIRECT_OUTPUT=False;
                print "Direct output set to: "+str(DIRECT_OUTPUT)
        if inp[0]=="last":
            if isRecording:
                lastRecordingTrack=True
                print "Recording the last track"
            else:
                inLoop=False

        if inp[0]=="level":
            if len(inp)==1:
                print "Current silence level value:"+str(SILENCE_LEVEL)    
            else:
                try:
                    SILENCE_LEVEL=int(inp[1])
                    print "New silence level value:"+str(SILENCE_LEVEL)
                except:
				    print "Parameter value could not be parsed"

thread.start_new_thread(update_params,())


p = pyaudio.PyAudio()

stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True, #this is for the direct sound output
                frames_per_buffer = chunk)




stepToStop=int(float(RATE/chunk) * SILENCE_SECONDS)
fadeoutSize=int(float(RATE/chunk) * FADEOUT_SECONDS)
stepsOfSilence=0
maxSignal=0
sampleLength=-1

print "Ready to work"

i=0

while inLoop:
    #we have just recived a new part of sound data
    data = stream.read(chunk)
    if DIRECT_OUTPUT:
        stream.write(data, chunk) #this wires the input to the output
    
    #let's see if it contains something loud enought
    summ=0;    
    for j in range(0,len(data)/2):
        summ=summ+abs(struct.unpack("h", data[2*j]+data[2*j+1])[0])
    
    lastSecondArray[i]=summ        #logging the sound volume 
    i=(i+1)%lastSecondFramesCount  #during the last second
                 
    if (((summ/chunk)>SILENCE_LEVEL) and (not(isRecording))):
        #if the signal is strong and we do not yet record, let's start
        WAVE_OUTPUT_FILENAME=strftime(FILENAME_FORMAT)+".wav";
        print WAVE_OUTPUT_FILENAME+" record started"
        all = []
        all.append(data)
        isRecording=True
        stepsOfSilence=0
        maxSignal=summ/chunk #storing the max signal value
        sampleLength=-1
    
    if (((summ/chunk)>SILENCE_LEVEL) and (isRecording)):
        #if we allready are recording, let's proceede
        all.append(data)
        stepsOfSilence=0
        sampleLength=-1
        
        if (summ/chunk)>maxSignal:  #storing the max signal value
            maxSignal=summ/chunk

    if (((summ/chunk)<=SILENCE_LEVEL) and (isRecording)):
        #if there is no more signal we do wait
        all.append(data)      
        if (sampleLength<0)and(stepsOfSilence==fadeoutSize):
            sampleLength=len(all)

        stepsOfSilence=stepsOfSilence+1
        
        if (stepsOfSilence>stepToStop):
            #oops, the silence was too long. it's over now
            isRecording=False
            if sampleLength<0:
                sampleLength=len(all)
            data = ''.join(all[0:sampleLength]) #the sample with no silence at the end 
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(data)
            wf.close()  
            all = []
            print "Record finished. File length: "+str(len(data)/(RATE*2))+" seconds"
            print "Max signal value: "+str(maxSignal) 
            if lastRecordingTrack:
                inLoop=False         

# the listening is over
        
if isRecording:
    #hey, we have some unfinished job
    isRecording=False
    if sampleLength<0:
        sampleLength=len(all)
    data = ''.join(all[0:sampleLength]) #the sample with no silence at the end 
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()  
    all = []
    print "Record finished. File length: "+str(len(data)/(RATE*2))+" seconds"          
    print "Max signal value: "+str(maxSignal) 

print "* done listening"

stream.close()
p.terminate()




