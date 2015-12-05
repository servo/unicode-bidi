def delete_all_lines_between_markers(filename, opening_marker, closing_marker):
    with open(filename, 'r') as file:
        data = file.readlines()
    i = 0
    remove_statement = False
    while i < len(data):
        v = data[i]
        #print("Here is ", v)
        if v.startswith(closing_marker):
            remove_statement = False
            #print(v, " : Completing Delete...")
        if remove_statement == True:
            #print("Removing", v)
            data.pop(i)
            i=i-1
        if v.startswith(opening_marker):
            #print(v, " : Starting Delete...")
            remove_statement = True
        i = i + 1
    with open(filename, 'w') as file:
        file.writelines(data)
    file.close()
#delete_all_lines_between_markers("target.txt", "//VickyStart", "//VickyEnd")

def remove_newline_char(str):
    return str.replace("\n", "")
def remove_newline_char_and_invalid_test_cases(l):
    i = 0
    while i < len(l):
        #print("before ", i,": ", l, end="")
        v = l[i]
        if v.startswith('\n') or v.startswith('\\\\') or v.startswith("#"):
            l.pop(i)
            i=i-1
        else:
            l[i] = remove_newline_char(l[i])
        #print("\t\tAfter ", i,": ", l)
        i = i + 1
    return l

def return_BidiCharacterTest_test_cases_in(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    for i in range(0, len(data)):
        data[i] = data[i] + "//-->BidiCharacterTest.txt Line Number:"+str(i+1)
    #print(" before remove_newline_char_and_invalid_test_cases", data)
    return remove_newline_char_and_invalid_test_cases(data)
#test_cases = return_BidiCharacterTest_test_cases_in("BidiCharacterTest.txt")

class BidiCharacterTestCase(object):
    def return_unicode_string(self, stringArr):
        s = ""
        for index in range(0, len(stringArr)):
            s = s + stringArr[index]
        return s
    def __init__(self, unformatted):
        field = unformatted.split(';')
        inp_arr = field[0].split(' ')
        try:
            oup_ind_and_line_marker = field[4].split("//-->")
        except IndexError:
            print("line which gave error: :", unformatted)
            print("field[4]:", field[4])
        #if len(oup_ind_and_line_marker) > 1:
    	self.marker = oup_ind_and_line_marker[1]
        oup_ind = oup_ind_and_line_marker[0].split(' ')
        oup_arr = []
        for index in range(0, len(inp_arr)):
            inp_arr[index] = "\\u{" + inp_arr[index] + "}"
        for index in range(0, len(oup_ind)):
            oup_arr.append(inp_arr[int(oup_ind[index])])
        self.inp = self.return_unicode_string(inp_arr)
        self.oup = self.return_unicode_string(oup_arr)
    def reorderline_assert_test(self):
        return "assert_eq!(reorder(\""+self.inp+"\"),\""+self.oup+"\");"+"//"+self.marker
#b = BidiCharacterTestCase("0061 0062 0063 0020 0028 0064 0065 0066 0020 0627 0628 062C 0029 0020 05D0 05D1 05D2;0;0;0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1;0 1 2 3 4 5 6 7 8 11 10 9 12 13 16 15 14//-->BidiCharacterTest.txt Line Number:2")
# b = BidiCharacterTestCase("05D0 2067 202A 0041;1;1;1 1 x 4;3 1 0//-->BidiCharacterTest.txt Line Number:2")
# print(b.inp)
# print(b.oup)
# print(b.reorderline_assert_test())
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
#insert_array_into_file_after_marker("target.txt", ["123\n", "456\n", "7890\n"], "//Vicky\n")

def parse_all_test_cases_from_BidiCharacterTest_txt():
	marker = "//BeginInsertedTestCases: Test cases from BidiCharacterTest.txt go here\n"
    #Read testcases from file BidiCharacterTest.txt and get an array from 'return_BidiCharacterTest_test_cases_in'
	unparsed_test_cases = return_BidiCharacterTest_test_cases_in("BidiCharacterTest.txt")
    #Parse each test case and derive input and output and Convert each test case to assert_reorder_line format: assert_eq!(reorder(a, b))
	BidiTestCaseList = []
	for testcase in unparsed_test_cases:
		BidiTestCaseList.append(BidiCharacterTestCase(testcase).reorderline_assert_test())
    #Write each test case in the list to output file after marker
	insert_list_into_file_after_marker("lib.rs", BidiTestCaseList, marker)