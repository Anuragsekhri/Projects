def getKeys(formatString):
    '''formatString is a format string with embedded dictionary keys.
    Return a set containing all the keys from the format string.'''

    List = list()
    end = 0
    repeat = formatString.count('{')
    for i in range(repeat):
        start = formatString.find('{', end) + 1
        end = formatString.find('}', start)
        key = formatString[start : end]
        List.append(key)

    return set(List)


def addPick(cue, dict):
    promptFormat = "Enter a specific example for {name}: "
    prompt = promptFormat.format(name=cue)
    response = input(prompt)
    dict[cue] = response


def getUserPicks(cues):
    userPicks = dict()
    for cue in cues:
        addPick(cue, userPicks)
    return userPicks

def tellStory(storyFormat):
    cues = getKeys(storyFormat)
    userPicks = getUserPicks(cues)
    story = storyFormat.format(**userPicks)
    print(story)

def main():
    originalStoryFormat = '''
Once upon a time, deep in an ancient jungle,
there lived a {animal}.  This {animal}
liked to eat {food}, but the jungle had
very little {food} to offer.  One day, an
explorer found the {animal} and discovered
it liked {food}.  The explorer took the
{animal} back to {city}, where it could
eat as much {food} as it wanted.  However,
the {animal} became homesick, so the
explorer brought it back to the jungle,
leaving a large supply of {food}.

The End
'''
    tellStory(originalStoryFormat)

main()