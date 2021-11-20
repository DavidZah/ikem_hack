import  musexmlex
import optparse
import xml

VERSION="none"
USAGE = "%prog [-h|-w|-c] xml_ecg_file.xml"
DESCRIPTION = ""
def process(filename, output):
    def start_element(name, attrs):
        g_Parser.start_element(name, attrs)
        # print ('Start element:', name, attrs)

    def end_element(name):
        g_Parser.end_element(name)
        # print ('End element:', name)

    def char_data(data):
        g_Parser.char_data(data)

    parser = optparse.OptionParser(usage=USAGE, version=VERSION, description=DESCRIPTION)

    #fileEncoding = options.encoding

    g_Parser = musexmlex.MuseXmlParser()

    p = xml.parsers.expat.ParserCreate()

    p.StartElementHandler = start_element
    p.EndElementHandler = end_element
    p.CharacterDataHandler = char_data


    f = musexmlex.codecs.open(filename, mode='r', encoding=None)
    print("Got here")
    p.Parse(f.read())
    g_Parser.makeZcg()
    g_Parser.writeCSV(output)


