import random

unicode_class_dict = dict()

# these are the surrogate codepoints, which are not valid rust characters
surrogate_codepoints = (0xd800, 0xdfff)

def is_surrogate(n):
    return surrogate_codepoints[0] <= n <= surrogate_codepoints[1]

def populate_unicode_class_data():
	filename = "UnicodeData.txt"
	line_list = []
	global unicode_class_dict
	with open(filename, 'r') as f:
		line_list = f.readlines()
	for line in line_list:
		field = line.split(';')
		hex_val = int(field[0], 16);
		if is_surrogate(hex_val):
			continue
		if field[4] in unicode_class_dict:
			unicode_class_dict[field[4]].append(field[0])
		else:
			unicode_class_dict[field[4]] = [field[0]]


def remove_newline_char(str):
    return str.replace("\n", "")
def remove_newline_char_and_invalid_test_cases(l):
    i = 0
    while i < len(l):
        #print("before ", i,": ", l, end="")
        v = l[i]
        if v.startswith('\n') or v.startswith('\\\\') or (v.startswith("#") and not v.startswith('#Count:')):
            l.pop(i)
            i=i-1
        else:
            l[i] = remove_newline_char(l[i])
            if l[i].startswith('#Count:'):
            	l[i] = l[i] + "//-->BidiTest.txt Line Number:"+str(i+1)
        #print("\t\tAfter ", i,": ", l)
        i = i + 1
    return l

def return_test_case_object_list_from(file_data2):
	test_case_objects = []
	levelsKeyword = '@Levels:'
	reorderKeyword = '@Reorder:'
	countKeyword = '#Count:'
	j = 0
	bidi_classes = []
	para_level = []
	while j < len(file_data2):
		if file_data2[j].startswith(levelsKeyword):
			levels = file_data2[j].lstrip(levelsKeyword).strip('\t').strip('\n').split(' ')
			#print levels
		elif file_data2[j].startswith(reorderKeyword): #@Reorder
			reorderVal = file_data2[j].lstrip(reorderKeyword).strip('\t').strip('\n').split(' ')
			#print reorderVal
		elif file_data2[j].startswith(countKeyword): #@Count
			countVal = file_data2[j].lstrip(countKeyword).strip('\t').strip('\n').split(' ')
			new_bidi_test_case = BidiTestCase(levels, reorderVal, bidi_classes, para_level)
			if len(new_bidi_test_case.levels)!=0:
				test_case_objects.append(new_bidi_test_case)
			#print("break at", j)
			bidi_classes = []
			para_level = []
		else:	#@ Paralevel
			data_fields = file_data2[j].strip('\n').strip('\t').split(';')
			bidi_classes.append(data_fields[0].strip('\n').split(' '))
			para_level.append(data_fields[1])
			#print para_level
		j = j + 1
	return test_case_objects


class BidiTestCase(object):
	def __init__(self, levels, reorderVal, bidi_classes, para_level):
		self.levels = levels
		self.reorderVal = reorderVal
		self.bidi_classes = bidi_classes
		#print(bidi_classes)
		self.para_level = para_level
		self.unicode_chars_list = self.replace_bidi_classes_with_random_chars(bidi_classes, levels)
		self.remove_wherever_x(levels, bidi_classes);
		self.levels =filter(lambda lev: lev != 'x', levels)

	def __repr__(self):
		return "\n--------------------------"+"\nlevels:"+str(self.levels) + "\nreorderVal:"+str(self.reorderVal) + "\nbidi_classes:"+str(self.bidi_classes) + "\npara_level:"+ str(self.para_level)+ "\nunicode_chars_list:"+ str(self.unicode_chars_list)+"\n--------------------------"

	#abcdef
	def str_rep(self, some_list):
		return str(some_list).replace("[", "").replace("]", "").replace(",", "").replace("'", "").replace(" ", "").replace("\\\\", "\\")
	#[a, b, c, d, e, f]
	def tupple_rep(self, some_list):
		return str(some_list).replace("'", "")
	#a, b, c, d, e, f
	def tupple_rep(self, some_list):
		return str(some_list).replace("'", "").replace("[", "").replace("]", "")

	def get_bidi_assert_test_case(self):
		return "assert_eq!(process_text(\""+self.str_rep(self.unicode_chars_list)+"\", Some("+self.reorderVal[0]+")), BidiInfo { levels: vec!["+self.tupple_rep(self.para_level)+"], classes: vec!["+self.tupple_rep(self.bidi_classes)+"], paragraphs: vec![ParagraphInfo { range: 0.."+str(len(self.bidi_classes))+", level: "+self.levels[0]+" } ], });"
	
	def replace_bidi_classes_with_random_chars(self, bidi_classes, levels):
		random_char_lists = []
		for bidi_class_list in bidi_classes:
			blist = []
			for index in range(0, len(bidi_class_list)):
				bidi_class_val = bidi_class_list[index]
				if levels[index] == 'x':
					continue
				blist.append("\u{"+random.choice(unicode_class_dict[bidi_class_val])+"}")
			random_char_lists.append(blist)
		return random_char_lists

	def remove_wherever_x(self, levels, bidi_classes):
		#return_lists = []
		list_num = 0
		for list_num in range(0, len(bidi_classes)):
			bidi_class_list = bidi_classes[list_num]
			index = 0
			while index < len(bidi_class_list):
				#print("before index: ", index, "levels: ", levels, "bidi_class_list: ", bidi_class_list)
				if levels[index] == 'x':
					bidi_class_list[index] = 'x';
				index = index + 1
				#print("after index: ", index, "levels: ", levels, "bidi_class_list: ", bidi_class_list)
			bidi_classes[list_num] = filter(lambda bid: bid != 'x', bidi_class_list)

def insert_list_into_file_after_marker(filename, array, marker):
    #Open File In read mode to read all lines
    with open(filename, 'r') as file:
        data = file.readlines()
    #Find 'marker' position
    insertPosition = data.index(marker) + 1
    #Insert lines into file
    for newLineNum in range(0, len(array)):
        data.insert(insertPosition+newLineNum, array[newLineNum]+"\n")
    #Write File and Close
    with open(filename, 'w') as file:
        file.writelines(data)
    file.close()

def fetch_BidiTest_txt_test_cases():
	#Populate Unicode Data from UnicodeData.txt
	populate_unicode_class_data()
	#Read test cases from BidiTest.txt
	filename = "BidiTest.txt"
	with open(filename, 'rt') as f:
		file_data = f.readlines()
	#clean the test cases
	file_data = list(file_data)
	file_data = remove_newline_char_and_invalid_test_cases(file_data)
	#convert test cases into rust assert test cases
	test_case_object_list = return_test_case_object_list_from(file_data)
	#collect all test cases
	test_case_array = []
	for tc in test_case_object_list:
	 	test_case_array.append(tc.get_bidi_assert_test_case())
	#write rust assert test cases to the file
	insert_list_into_file_after_marker("lib.rs", test_case_array, "//BeginInsertedTestCases: Test cases from BidiTest.txt go here\n")

#fetch_BidiTest_txt_test_cases()
# a = ["abc", "def", "ghi", "jkl"]
# print()