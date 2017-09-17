# -*- coding: utf8 -*-

import preprocessor as p
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
i=0

text_file = open("trumptweetsprocessed.txt", "w")
p.set_options(p.OPT.URL, p.OPT.EMOJI)
with open('trumptweets.txt', 'r') as f:
	for line in f:
		line=(p.clean(line))
		text_file.write(line)
		text_file.write("\n")




text_file.close()