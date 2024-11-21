from art import *


for font in FONT_NAMES:
    print(font)
    #tprint("HaikuGRU", font=font)
    #tprint("Created by: \n E. Banerjee\n D.C.R Escalante\n N.A. Monti\n J.H. Sayo", font=font)
    tprint("Would you like to skip training? (Y/n): ", font=font)




#tprint("Welcome to: HaikuGRU", font="sub-zero")
#tprint("Created by: N. Monti", font="slant")

#tprint("Enter your prompt: ", font="bell")
#prompt = input("")

#tprint(prompt, font="smslant")

def start_menu():
    tprint("Welcome to: HaikuGRU", font="sub-zero")
    tprint("Created by: \n E. Banerjee\n D.C.R Escalante\n N.A. Monti\n J.H. Sayo", font="slant")

    tprint("Would you like to skip training? (Y\\n): ", font="bell")
    prompt = input("")

    prompt = prompt.lower()

    if prompt == "y":
        prompt = True
    else:
        prompt = False

    return prompt