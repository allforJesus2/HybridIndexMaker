import re

#this is the main correct_fn
def correct(result_words):
    # remove numbers from tag
    #box, text, score = result
    words = result_text.split()

    if len(words) > 2:
        for i, word in enumerate(words):
            if len(word) > 2:
        tag, tag_no = words[0], words[1]

    if len(words) == 1:
        tag, tag_no = words[0], ''

    if len(words) == 3:
        tag, tag_no = words[0], words[1]

    tag = replace_numbers_with_letters(tag)
    tag_no = correct_tag_no(tag_no)
    return tag, tag_no
    # remove certain letters from tag no

def correct_tag_no(tag_no):
    # this example captures starting at 9XWWW
    pattern = r'8\d{1}\w{3}'
    match = re.findall(pattern, tag_no)
    if match:
        # Extract the first 4 digits and the optional letter
        tag_no_match = match[0]
        last_char = tag_no_match[-1]
        if last_char.isdigit():
            last_char = replace_numbers_with_letters(last_char)
        tag_no_corrected = tag_no_match[:-1]+last_char
        print(f'tag_no_corrected from {tag_no} to {tag_no_corrected}')
        return tag_no_corrected
    else:
        return tag_no


def replace_numbers_with_letters(string):
    replacements = {
        '0': 'O',
        '1': 'I',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
    }

    replaced_string = ""
    for char in string:
        if char in replacements:
            replaced_string += replacements[char]
            print(f"made correction")
        else:
            replaced_string += char

    return replaced_string
