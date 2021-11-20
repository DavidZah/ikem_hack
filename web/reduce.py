import re

def is_regular_char(char):
    not_edge_characters = ",./;)(*!@#$%^&|=-?:+><•"
    try:
        n = not_edge_characters.index(char)
        return False
    except ValueError:
        return True


def reduce(text):
    buffer = []
    for word in text.split(" "):
        w = one(word)
        if w != "":
            buffer.append(w)
    return " ".join(buffer)


def one(text):
    improved = regular_strip(text.lower())
    if is_date(improved):
        return ""
    else:
        return improved


def regular_strip(text):
    start = 0
    for pos in range(len(text)):
        if is_regular_char(text[pos]):
            break
        start = pos+1

    end = len(text)
    for pos in range(len(text)-1, 0, -1):
        if pos <= start:
            break
        if is_regular_char(text[pos]):
            break
        end = pos
    return text[start:end]


def is_date(str):
    m = re.search("^[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}$", str)
    if m:
        return True
    else:
        return False


#print(is_date("15.1.202"))

