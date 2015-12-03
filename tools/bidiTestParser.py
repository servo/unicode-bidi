import os
import random
def fetch_BidiTestCases_from_file():
    #filename = os.path.join('../src', 'BidiTest.txt') 
    filename = "BidiTest.txt"
    with open(filename, 'rt') as f:
    	file_data = f.readlines()
    file_data = list(file_data)
    return remove_comments(file_data)

def remove_comments(fl_data):
	i = 0
	while i < len(fl_data):
		remove_comm = fl_data[i]
		if (remove_comm.startswith('#') and not remove_comm.startswith('#Count:')) or remove_comm.startswith("\n"):
			fl_data.pop(i)
			i = i - 1
		else:
			fl_data[i] = remove_new_line(fl_data[i])
		i = i + 1
	return seperate_parse_data(fl_data)

def remove_new_line(rem_line):
	new_file_data = rem_line.replace("\n", "")
	return new_file_data

def seperate_parse_data(file_data2):
	levelsKeyword = '@Levels:'
	reorderKeyword = '@Reorder:'
	countKeyword = '#Count:'
	j = 0
	test_cases = []
	while j < len(file_data2):
		if file_data2[j].startswith(levelsKeyword):
			levels = file_data2[j].lstrip(levelsKeyword).strip('\t').strip('\n').split(' ')
			#print levels
		elif file_data2[j].startswith(reorderKeyword):
			reorderVal = file_data2[j].lstrip(reorderKeyword).strip('\t').strip('\n').split(' ')
			#print reorderVal
		elif file_data2[j].startswith(countKeyword):
			countVal = file_data2[j].lstrip(countKeyword).strip('\t').strip('\n').split(' ')
			generate_final_output(levels,reorderVal,test_cases, para_level)
		else:
			data_fields = file_data2[j].strip('\n').strip('\t').split(';')
			test_cases.append(data_fields[0].strip('\n').split(' '))
			para_level = data_fields[1]
			#print para_level
		j = j + 1

def generate_final_output(levels, reorderVal, test_cases, para_level):
	i = 0
	j = 0 
	ans = []
	final_oup = []
	while i < len(test_cases):
		#print test_cases[i]
		for j in range(0, len(test_cases[i])):
			ans.append(generate_random_character(test_cases[i][j]))
			j  = j + 1
		final_oup.append(insert_into_process_text(levels, reorderVal, ans, test_cases[i],para_level))
		del ans[:]
		i = i+1
	#Insertion process begins
	opening_marker = "//BeginInsertedTestCases: Test cases from BidiTest.txt go here\n"
	closing_marker = "//EndInsertedTestCases: Test cases from BidiTest.txt go here\n"
	#filename = os.path.join('../src', 'lib.rs') 
	filename = "lib.rs"
	#delete_all_lines_between_markers(filename, opening_marker, closing_marker)
	with open(filename, 'r') as file:
		data = file.readlines()
	#Find 'marker' position
	insertPosition = data.index(opening_marker) + 1
	#Insert lines into file
	for newLineNum in range(0, len(final_oup)):
		data.insert(insertPosition+newLineNum, final_oup[newLineNum]+"\n")
	#Write File and Close
	with open(filename, 'w') as file:
		file.writelines(data)
	file.close()


def generate_random_character(test_cases):
	#filename = os.path.join('../src', 'UnicodeData.txt')
	filename = "UnicodeData.txt"
	unicode1 = []
	with open(filename, 'rt') as f:
		char_data = f.readlines()
	char_data = list(char_data)	
	i = 0
	while i < len(char_data):
		check_data = char_data[i].split(';')
		if test_cases in check_data[4]:
			 unicode1.append(check_data[0])
		i = i+1
	return random.choice(unicode1)

def return_unicode_string(input_text):
	s = ""
	for i in range(0, len(input_text)):
		s = s + input_text[i]
	return s

def return_array(input_text):
	s = ""
	for i in range(0, len(input_text)-1):
		s = s + input_text[i] + ","
	s = s + input_text[len(input_text)-1]
	return s

def insert_into_process_text(levels, reorderVal, ans, bidi_classes,para_level):

	#print "assert_eq!(process_text(, None), BidiInfo {levels: vec![],classes: vec![] ,paragraphs: vec![ParagraphInfo {}],});"
	for i in range(0, len(ans)):
		ans[i] = '\\'+"u{" + ans[i] + "}"	
	input_text_final = return_unicode_string(ans)
	levels_final = return_array(levels)
	bidi_class_final = return_array(bidi_classes)
	range_final = str(len(bidi_classes))
	return "assert_eq!(process_text(\""+input_text_final+"\", None), BidiInfo { levels:  vec!["+levels_final+"], classes: vec!["+bidi_class_final+"],paragraphs: vec![ParagraphInfo { range: 0.."+range_final+", level: "+para_level+" }],});"

def delete_all_lines_between_markers(filename, opening_marker, closing_marker):
    with open(filename, 'r') as file:
        data = file.readlines()
    i = 0
    remove_statement = False
    while i < len(data):
        v = data[i]
        if v.startswith(closing_marker):
            remove_statement = False
            #print(v, "Found")
        if remove_statement == True:
            #print("Removing", v)
            data.pop(i)
            i=i-1
        if v.startswith(opening_marker):
            #print(v, "Found")
            remove_statement = True
        i = i + 1
    with open(filename, 'w') as file:
        file.writelines(data)
    file.close()

#fetch_BidiTestCases_from_file()