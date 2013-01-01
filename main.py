# Main module of MSD
# Date: 18.08.2012
# Todo:
# 

# Standard Python imports
import sys
import re

# Third Party imports

def readData(fname):
    inputFile = open(fname, "r")
    result = []
    for line in inputFile:
        result.append(re.sub(" +|\t", " ", line).strip().split(" "))
    return result

def main():
    pass
    temp = readData("../data/taste_profile_song_to_tracks.txt")
    print len(temp)



if __name__ == "__main__":
    main()
